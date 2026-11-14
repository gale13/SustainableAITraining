"""Run the same resumable study in Docker or in a user-owned Python environment."""
import fcntl
from importlib import metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from .common import atomic_json, digest, file_sha256, read_json


PROJECT = Path(__file__).resolve().parents[1]
NATIVE = "PROFILE_NATIVE_ROOT" in os.environ
WORK = Path(os.environ["PROFILE_NATIVE_ROOT"]).resolve() if NATIVE else Path("/")


def log(message):
    path = WORK / "results" / os.environ["PROFILE_RUN_ID"] / "coordinator.log"
    with path.open("a", encoding="utf-8") as stream:
        print(message, file=stream)
    print(message, flush=True)


def module(name, *args):
    command = [sys.executable, "-m", "profiling." + name, *map(str, args)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            log(line.rstrip("\n"))
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def prepare(data_root):
    dataset = data_root / "wt103-raw-first32k-s512-v1"
    with (data_root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not dataset.exists():
            with tempfile.TemporaryDirectory(dir=data_root, prefix=".preparing-") as temp:
                staging = Path(temp) / "prepared"
                module("prepare", "--out", staging, "--vocab-size", 32768, "--vocab-source", "first-block")
                staging.rename(dataset)
        manifest = read_json(dataset / "manifest.json")
        if (manifest["config"], manifest["vocab_source"], manifest["vocab_size"], manifest["seq_len"]) != (
            "wikitext-103-raw-v1", "first-block", 32768, 512
        ):
            raise ValueError(f"Unexpected dataset or vocabulary in {dataset}")
    return dataset


def status(root, stage, **fields):
    atomic_json(root / "status.json", {"stage": stage, "unix_time": time.time(), **fields})
    log(f"[{root.name}] {stage}")


def execute(root):
    settings = root / "job.json"
    if NATIVE:
        runtime = {"python": sys.version, "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
                   "source": {p.name: file_sha256(p) for p in sorted((PROJECT / "profiling").glob("*.py"))}}
        identity = {"native_runtime_sha256": digest(runtime)}
    else:
        identity = {"image_id": os.environ["PROFILE_IMAGE_ID"]}
    if settings.exists():
        job = read_json(settings)
        if any(job.get(key) != value for key, value in identity.items()):
            raise ValueError("Runtime differs from this job's original environment. Restore it or start a new run ID.")
    else:
        job = {"config": read_json(PROJECT / "configs/default.json"), "started_unix": time.time(), **identity}
        if NATIVE:
            atomic_json(root / "native_environment.json", runtime)
        atomic_json(settings, job)
    config = root / "config.json"
    atomic_json(config, job["config"])
    deadline = job["started_unix"] + job["config"]["budget_hours"] * 3600
    status(root, "preflight")
    module("launch", "preflight", "--eight-5090s")
    status(root, "preparing_data")
    data = prepare(WORK / "data")
    shared = ["--data", data, "--config", config, "--deadline-unix", deadline]
    plan = root / "plan.json"
    if not plan.exists():
        status(root, "calibrating")
        pilots = root / "pilot"
        pilots.mkdir(exist_ok=True)
        # A fresh attempt after interruption preserves the partial measurements.
        attempt = pilots / str(time.time_ns())
        module("launch", "pilot", *shared, "--out", attempt)
        calibrated = read_json(attempt / "plan.json")
        calibrated["pilot_attempt"] = attempt.name
        atomic_json(plan, calibrated)
    formal = root / "training"
    status(root, "training")
    module("launch", "run", *shared, "--plan", plan, "--out", formal)
    progress = read_json(formal / "progress.json")
    if not progress["common_completed_blocks"]:
        raise RuntimeError("Budget expired before the first common block. Use a new run ID for a fresh budget.")
    status(root, "auditing")
    module("audit", formal)
    audit = read_json(formal / "audit.json")
    # Only checkpoints created by this job are temporary. Preserve every measurement.
    for batch in job["config"]["batches"]:
        state = formal / f"batch_{batch:03d}" / "_state"
        if state.exists():
            shutil.rmtree(state)
    stage = "complete" if audit["all_duration_targets_met"] else "complete_with_duration_shortfall"
    status(root, stage, common_completed_blocks=progress["common_completed_blocks"],
           epochs_per_block=audit["epochs_per_block"], stop_reason=progress["stop_reason"],
           elapsed_hours=(time.time() - job["started_unix"]) / 3600)


def main():
    root = WORK / "results" / os.environ["PROFILE_RUN_ID"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if old.get("stage") not in ("complete", "complete_with_duration_shortfall"):
            try:
                # Cover the race between checking for other GPU processes and spawning workers.
                with (root.parent / ".eight-gpu.lock").open("a") as devices_lock:
                    try:
                        fcntl.flock(devices_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise RuntimeError("Another active run in this workspace holds the eight GPUs") from exc
                    execute(root)
            except Exception as exc:
                log(traceback.format_exc())
                status(root, "failed", error=str(exc), previous_stage=read_json(root / "status.json")["stage"]
                       if (root / "status.json").exists() else "initialization")
                if NATIVE:
                    raise SystemExit(1) from exc
    if not NATIVE:
        os.execlp("sleep", "sleep", "infinity")


if __name__ == "__main__":
    main()
