"""Coordinate independent single-GPU workers, without DDP or gradient sharing."""
import argparse
import contextlib
import fcntl
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from .common import atomic_json, digest, file_sha256, make_plan, read_json
from .energy import inventory, Sensor


def entry(connection, config, manifest, data, output, gpu, batch, mode, plan, committed, fingerprint):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    os.environ["OMP_NUM_THREADS"] = str(config["cpu_threads_per_worker"])
    os.environ["MKL_NUM_THREADS"] = str(config["cpu_threads_per_worker"])
    log_path = Path(output) / f"batch_{batch:03d}" / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            from .engine import worker
            worker(connection, config, manifest, data, output, gpu, batch, mode, plan, committed, fingerprint)
        except BaseException:
            detail = traceback.format_exc()
            print(detail, flush=True)
            try:
                connection.send({"event": "error", "batch": batch, "detail": detail})
            except (BrokenPipeError, EOFError):
                pass
            raise


def receive_all(connections, processes, event):
    pending, results = set(connections), {}
    while pending:
        for conn in wait(list(pending), timeout=1):
            try:
                message = conn.recv()
            except EOFError as exc:
                raise RuntimeError("Worker exited before completing the common block; inspect worker.log") from exc
            if message["event"] == "error":
                raise RuntimeError(message["detail"])
            if message["event"] != event:
                raise RuntimeError(f"Unexpected worker message: {message}")
            pending.remove(conn)
            results[conn] = message
        if any(p.exitcode is not None and p.exitcode != 0 for p in processes):
            raise RuntimeError("GPU worker failed; all remaining workers will be stopped")
    return list(results.values())


def validate_data(data, manifest, config):
    if config["model"]["seq_len"] != manifest["seq_len"]:
        raise ValueError("Configuration and prepared sequence lengths differ")
    for name, expected in [("train.bin", manifest["train"]["sha256"]),
                           ("validation.bin", manifest["validation"]["sha256"]),
                           ("vocab.json", manifest["vocab_sha256"])]:
        if file_sha256(Path(data) / name) != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")


def check_exclusive(gpus):
    for gpu in gpus:
        query = subprocess.run(["nvidia-smi", "-i", gpu["uuid"], "--query-compute-apps=gpu_uuid,pid",
                                "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
        if any(line.split(",")[0].strip() == gpu["uuid"] for line in query.stdout.splitlines()):
            raise RuntimeError(f"GPU {gpu['uuid']} already has compute processes. Reserve empty GPUs before starting.")


def validate_config(config):
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if len(config["batches"]) != len(set(config["batches"])) or any(b <= 0 for b in config["batches"]):
        raise ValueError("Batches must be positive and distinct")
    for key in ["eval_batch", "cpu_threads_per_worker", "pilot_seconds", "pilot_min_epochs", "power_interval_seconds",
                "minimum_block_seconds", "budget_hours", "learning_rate", "grad_clip"]:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < config["memory_fraction_limit"] < 1 or config["runtime_margin"] < 1 or config["minimum_duration_margin"] < 1:
        raise ValueError("Invalid memory or timing margin")
    if config.get("warmup_steps", config.get("warmup_tokens", 0)) < 0 or config["warmup_seconds"] < 0:
        raise ValueError("Warmup values cannot be negative")


def execute(args):
    config = read_json(args.config)
    validate_config(config)
    manifest = read_json(Path(args.data) / "manifest.json")
    validate_data(args.data, manifest, config)
    # The probe must exit before checking for other GPU processes or measuring board power.
    devices = json.loads(subprocess.check_output([sys.executable, "-m", "profiling.launch", "devices"], text=True))
    if args.gpus:
        indices = [int(s) for s in args.gpus.split(",")]
    else:
        indices = [g["index"] for g in devices[:len(config["batches"])]]
    if len(set(indices)) != len(indices) or len(indices) != len(config["batches"]):
        raise ValueError("Choose exactly one distinct GPU per configured batch")
    by_index = {gpu["index"]: gpu for gpu in devices}
    if any(i not in by_index for i in indices):
        raise ValueError("Requested GPU is outside this job's CUDA allocation")
    gpus = [by_index[i] for i in indices]
    check_exclusive(gpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.mode == "pilot" and (out / "run.json").exists():
        raise ValueError("Use a new output directory for every pilot")
    plan = read_json(args.plan) if args.mode == "run" else None
    if plan:
        if plan["config_sha256"] != digest(config) or plan["manifest_sha256"] != digest(manifest):
            raise ValueError("Plan/configuration/data mismatch; rerun calibration")
        if [g["uuid"] for g in gpus] != plan["gpu_uuids"]:
            raise ValueError("Pilot and run must use the same GPU-to-batch mapping")
        if not (1 <= plan["planned_blocks"] <= len(manifest["blocks"])) or plan["epochs_per_block"] < 1:
            raise ValueError("Invalid epoch or block count")
    fingerprint = digest({"config": config, "manifest": manifest, "plan": plan, "gpu_uuids": [g["uuid"] for g in gpus]})
    committed = 0
    if (out / "run.json").exists():
        old = read_json(out / "run.json")
        if old["fingerprint"] != fingerprint:
            raise ValueError("Existing run has different settings; use a new output directory")
        progress = read_json(out / "progress.json")
        committed = progress["common_completed_blocks"]
        if progress.get("finished"):
            print("This run is already finished; outputs left unchanged.")
            return
    else:
        progress = {"common_completed_blocks": 0, "finished": False,
                    "deadline_unix": time.time() + config["budget_hours"] * 3600,
                    "last_block_wall_seconds": None, "duration_violations": []}
        if args.deadline_unix is not None:
            progress["deadline_unix"] = args.deadline_unix
        atomic_json(out / "progress.json", progress)
        atomic_json(out / "config.json", config)
        atomic_json(out / "data_manifest.json", manifest)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (out / "environment-pip-freeze.txt").write_text(freeze.stdout)
        atomic_json(out / "run.json", {"fingerprint": fingerprint, "config": config, "manifest": manifest,
                                      "plan": plan, "gpus": gpus, "platform": platform.platform(),
                                      "python": sys.version, "created_unix": time.time(), "mode": args.mode})
    context = mp.get_context("spawn")
    connections, processes = [], []
    try:
        for gpu, batch in zip(gpus, config["batches"]):
            parent, child = context.Pipe()
            process = context.Process(target=entry, args=(child, config, manifest, args.data, str(out), gpu,
                                                          batch, args.mode, plan, committed, fingerprint))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        ready = receive_all(connections, processes, "ready")
        if len({r["initial_hash"] for r in ready}) != 1:
            raise RuntimeError("Workers did not start from identical parameter tensors")
        limit = plan["planned_blocks"] if plan else 1
        reason = "planned_blocks_complete"
        for block_id in range(committed, limit):
            predicted = max(plan["predicted_slowest_block_seconds_with_margin"],
                            (progress["last_block_wall_seconds"] or 0) * config["runtime_margin"]) if plan else 0
            if plan and time.time() + predicted > progress["deadline_unix"] - config["budget_reserve_seconds"]:
                reason = "insufficient_budget_for_another_complete_common_block"
                break
            start_at = time.perf_counter() + 2
            for conn in connections:
                conn.send({"action": "block", "block_id": block_id, "start_at": start_at})
            done = receive_all(connections, processes, "done")
            if any(r["block_id"] != block_id for r in done):
                raise RuntimeError("Worker block IDs differ")
            progress["common_completed_blocks"] = block_id + 1
            progress["last_block_wall_seconds"] = max(r["block_wall_seconds"] for r in done)
            if plan and any(r["train_block_seconds"] < config["minimum_block_seconds"] for r in done):
                progress["duration_violations"].append(block_id)
                print(f"WARNING: block {block_id} was shorter than the target on at least one GPU; epochs remain fixed.", flush=True)
            # This is the transaction commit. On restart every worker reloads this same block boundary.
            atomic_json(out / "progress.json", progress)
            print(f"All GPUs completed block {block_id}; max wall time {progress['last_block_wall_seconds']:.1f}s", flush=True)
        for conn in connections:
            conn.send({"action": "stop"})
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError("Worker did not exit cleanly")
        if args.mode == "pilot":
            pilots = [read_json(out / f"batch_{b:03d}" / "pilot.json") for b in config["batches"]]
            elapsed = (max(0, time.time() - (args.deadline_unix - config["budget_hours"] * 3600))
                       if args.deadline_unix is not None else 0)
            plan = make_plan(config, manifest, pilots, elapsed)
            plan["gpu_uuids"] = [g["uuid"] for g in gpus]
            atomic_json(out / "plan.json", plan)
            print(f"Plan: {plan['epochs_per_block']} epochs/block, {plan['planned_blocks']} common blocks, "
                  f"estimated {plan['predicted_total_hours_with_margin']:.2f}h with margin", flush=True)
        progress.update(finished=True, stop_reason=reason, finished_unix=time.time())
        atomic_json(out / "progress.json", progress)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for conn in connections:
            conn.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("devices", help="List CUDA-visible devices and their matching telemetry UUIDs")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--eight-5090s", action="store_true", help="Check the target study hardware")
    for mode in ("pilot", "run"):
        p = sub.add_parser(mode)
        p.add_argument("--config", default="configs/default.json")
        p.add_argument("--data", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--deadline-unix", type=float, help="Shared deadline including preparation and pilot")
        p.add_argument("--gpus", help="Physical nvidia-smi indices within the allocation; default: one allocated GPU per batch in index order")
        if mode == "run":
            p.add_argument("--plan", required=True)
    args = parser.parse_args()
    if args.mode == "devices":
        print(json.dumps(inventory()))
    elif args.mode == "preflight":
        import torch
        print(f"torch={torch.__version__}, CUDA={torch.version.cuda}, arch={torch.cuda.get_arch_list()}")
        devices = inventory()
        if args.eight_5090s and (len(devices) != 8 or len({g["name"] for g in devices}) != 1
                                or any("5090" not in g["name"] for g in devices)):
            raise RuntimeError("The eight-GPU study requires eight visible RTX 5090s of the same model")
        for gpu in devices:
            sensor = Sensor(gpu["uuid"])
            try:
                print(gpu, sensor.read(), sensor.init_note)
            finally:
                sensor.close()
        print("Preflight telemetry OK. The concurrent pilot tests actual model forward/backward kernels and memory.")
    else:
        execute(args)


if __name__ == "__main__":
    main()
