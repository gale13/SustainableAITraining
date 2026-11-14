import os
import sys
from pathlib import Path
from unittest.mock import patch

base = Path.cwd()
sys.path[:0] = [str(base), str(base / "scripts")]
import diagnose_nonfinite as diag

run = diag.read(base / "results/wt103-01/training/run.json")
block = 16
checkpoint = base / "results/wt103-01/training/batch_004/_state/checkpoint_0016.pt"
assert diag.read(base / "results/wt103-01/training/progress.json")["common_completed_blocks"] == block
import fcntl
lock = (base / "results/.eight-gpu.lock").open("rb")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def trace(engine, tokens, block, epoch, train):
    import torch
    if epoch > 3:
        print("Three diagnostic epochs completed", flush=True)
        sys.exit(0)
    counts = [0, 0]
    @torch.no_grad()
    def clip(params, limit):
        params = list(params)
        grads = [p.grad for p in params if p.grad is not None]
        norm = torch.nn.utils.get_total_norm(grads)
        counts[0] += 1
        if not torch.isfinite(norm):
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(g, dtype=torch.float64) for g in grads
            ]))
            if not torch.isfinite(norm):
                bad = [name for name,p in engine.model.named_parameters()
                       if p.grad is not None and not torch.isfinite(p.grad).all()]
                raise FloatingPointError(f"epoch={epoch} step={counts[0]} nonfinite gradient {bad}")
            if counts[1] == 0:
                print(f"epoch={epoch} step={counts[0]} fp64_norm={norm.item():.6g}", flush=True)
            counts[1] += 1
        torch.nn.utils.clip_grads_with_norm_(params, limit, norm)
    with patch.object(torch.nn.utils, "clip_grad_norm_", clip):
        result = engine.epoch(tokens, train=train)[0]
    if train:
        print(f"epoch={epoch} fallback={counts[1]}/{counts[0]}", flush=True)
    return result

diag.trace_epoch = trace
diag.replay(run, checkpoint, block, 4, base)
