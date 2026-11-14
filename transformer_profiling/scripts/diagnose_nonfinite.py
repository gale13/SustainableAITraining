"""Inspect a stopped formal run; optionally replay its next block without writing any outputs."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


def read(path):
    return json.loads(path.read_text())


def check_state(file, run, block, batch):
    import torch
    state = torch.load(file, map_location="cpu", weights_only=False)  # This run's own checkpoint only.
    target = run["manifest"]["block_target_tokens"]
    epochs = run["plan"]["epochs_per_block"] * block
    seq = run["config"]["model"]["seq_len"]
    if (state["completed_blocks"] != block or state["fingerprint"] != run["fingerprint"]
            or state["tokens_seen"] != epochs * target or state["steps"] != epochs * math.ceil((target // seq) / batch)):
        raise ValueError(f"Inconsistent progress/fingerprint in {file}")
    for name, value in state["model"].items():
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{file}: nonfinite model tensor {name}")
    for param, slots in state["optimizer"]["state"].items():
        for name, value in slots.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                raise ValueError(f"{file}: nonfinite Adam tensor {param}.{name}")
            if name == "exp_avg_sq" and (value < 0).any().item():
                raise ValueError(f"{file}: negative Adam second moment for parameter {param}")
    print(f"batch={batch} checkpoint finite, global_steps={state['steps']}", flush=True)


def trace_epoch(engine, tokens, block, epoch, train):
    import torch
    from torch.nn import functional as F
    original_loss, original_clip = F.cross_entropy, torch.nn.utils.clip_grad_norm_
    phase = "train" if train else "validation"
    step_in_epoch = 0

    def checked_loss(logits, targets, *a, **kw):
        nonlocal step_in_epoch
        step_in_epoch += 1
        loss = original_loss(logits, targets, *a, **kw)
        if not torch.isfinite(loss).item():
            logits_finite = torch.isfinite(logits).all().item()
            raise FloatingPointError(f"cross_entropy={loss.item()}, logits_finite={logits_finite}, "
                                     f"target_range={targets.min().item()}..{targets.max().item()}")
        return loss

    def checked_clip(params, *a, **kw):
        try:
            kw["error_if_nonfinite"] = True
            return original_clip(params, *a, **kw)
        except RuntimeError as exc:
            bad = [name for name, p in engine.model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all().item()]
            raise RuntimeError(f"gradient clipping failed; nonfinite gradient tensors={bad}; {exc}") from exc

    with patch.object(F, "cross_entropy", checked_loss), patch.object(torch.nn.utils, "clip_grad_norm_", checked_clip):
        try:
            return engine.epoch(tokens, train=train)[0]
        except Exception:
            print(f"REPLAY FAILURE block={block} epoch={epoch} phase={phase} step_in_epoch={step_in_epoch} "
                  f"completed_global_steps={engine.steps} lr={engine.optimizer.param_groups[0]['lr']}", flush=True)
            raise


def replay(run, file, block, batch, project):
    # A separate process discovers the allocation before this process initializes CUDA.
    visible = json.loads(subprocess.check_output(
        [sys.executable, "-m", "profiling.launch", "devices"], cwd=project, text=True))
    uuid = run["gpus"][run["config"]["batches"].index(batch)]["uuid"]
    if uuid not in [gpu["uuid"] for gpu in visible]:
        raise RuntimeError("Original GPU is outside the current allocation")
    from profiling.launch import check_exclusive, validate_data
    check_exclusive([{"uuid": uuid}])
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    import torch
    from profiling.engine import Engine
    torch.set_num_threads(run["config"]["cpu_threads_per_worker"])
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() != 1 or ("GPU-" + str(torch.cuda.get_device_properties(0).uuid)).lower() != uuid.lower():
        raise RuntimeError("Replay process must see exactly the original GPU")
    data = project / "data" / "wt103-raw-first32k-s512-v1"
    validate_data(data, run["manifest"], run["config"])
    engine = Engine(run["config"], run["manifest"], data, batch)
    if engine.restore(file, run["fingerprint"]) != block:
        raise ValueError("Checkpoint block mismatch")
    tokens = engine.block(block)
    print(f"Replay only: block={block}, batch={batch}, uuid={uuid}", flush=True)
    for epoch in range(1, run["plan"]["epochs_per_block"] + 1):
        loss = trace_epoch(engine, tokens, block, epoch, True)
        val = trace_epoch(engine, engine.validation, block, epoch, False)
        print(f"epoch={epoch}, train_loss={loss:.6f}, validation_loss={val:.6f}", flush=True)
    print("Original failure did not reproduce in this replay.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="results/wt103-01")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--replay", action="store_true", help="Use the original single GPU for one diagnostic block")
    args = parser.parse_args()
    root = args.run.resolve()
    formal = root / "training"
    project = Path.cwd()
    sys.path.insert(0, str(project))
    sys.dont_write_bytecode = True
    lock = root.parent / ".eight-gpu.lock" if args.replay else root / ".lock"
    with lock.open("rb") as handle:
        fcntl.flock(handle, (fcntl.LOCK_EX if args.replay else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        run, progress = read(formal / "run.json"), read(formal / "progress.json")
        block = progress["common_completed_blocks"]
        if args.batch not in run["config"]["batches"]:
            parser.error("Unknown batch")
        groups = formal / f"batch_{args.batch:03d}"
        name = f"checkpoint_{block:04d}.pt"
        plan, config = run["plan"], run["config"]
        pred = max(plan["predicted_slowest_block_seconds_with_margin"],
                   (progress["last_block_wall_seconds"] or 0) * config["runtime_margin"])
        latest = progress["deadline_unix"] - config["budget_reserve_seconds"] - pred
        zone = timezone(timedelta(hours=8))
        print(f"committed={block}, planned={plan['planned_blocks']}, epochs/block={plan['epochs_per_block']}, "
              f"lr={config['learning_rate']}, precision={config['precision']}", flush=True)
        print("Latest budget check that can admit one more block (Beijing):",
              datetime.fromtimestamp(latest, zone).isoformat(timespec="seconds"), flush=True)
        for attempt in sorted((groups / "attempts").glob(f"block_{block:04d}_*.csv")):
            with attempt.open() as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                print(attempt.name, [(r["epoch_in_block"], r["train_loss"], r["validation_loss"]) for r in rows[-3:]], flush=True)
        if args.replay:
            if progress["finished"] or block >= plan["planned_blocks"]:
                raise ValueError("No pending formal block")
            replay(run, groups / "_state" / name, block, args.batch, project)
        else:
            import torch
            torch.set_num_threads(1)
            for batch in config["batches"]:
                check_state(formal / f"batch_{batch:03d}" / "_state" / name, run, block, batch)


if __name__ == "__main__":
    main()
