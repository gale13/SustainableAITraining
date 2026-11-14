"""One-time migration of the stopped, verified wt103-01 run at committed block 16."""
import fcntl
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path.cwd()))
from profiling.common import atomic_json, digest, file_sha256, read_json

root = Path("results/wt103-01")
backup = Path.home() / "wt103-recovery/wt103-01-c16"
formal = root / "training"
locks = []
for path in [root / ".nohup.lock", root / ".lock", root.parent / ".eight-gpu.lock", formal / ".lock"]:
    lock = path.open("rb")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(lock)
job = read_json(root / "job.json")
old = read_json(root / "native_environment.json")
progress = read_json(formal / "progress.json")
status = read_json(root / "status.json")
assert status["stage"] == "failed" and progress == read_json(backup / "progress.json")
assert progress["common_completed_blocks"] == 16 and not progress["finished"]
assert "Five diagnostic epochs completed" in (backup / "math-probe.log").read_text()
assert not (root / "recovery_c16.json").exists()

def runtime():
    return {"python": sys.version, "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
            "source": {p.name: file_sha256(p) for p in sorted(Path("profiling").glob("*.py"))}}

assert runtime() == old and digest(old) == job["native_runtime_sha256"]
patched = dict(old["source"], **{
    "model.py": "dc61bde1077d1e8b17d2d2702e439789a1df78056abbc0646c8753763b811895",
    "engine.py": "1bdf2af7e4e9f259aa3d27f577dcda7bcba8f8edeea9edbed17a066ad800dc2f"})
remaining = progress["deadline_unix"] - status["unix_time"]
assert 0 < remaining < job["config"]["budget_hours"] * 3600
for src in [root / "job.json", root / "native_environment.json", Path("profiling/engine.py"), Path("profiling/model.py")]:
    dst = backup / src.name
    if dst.exists():
        assert file_sha256(src) == file_sha256(dst)
    else:
        shutil.copy2(src, dst)
for batch in job["config"]["batches"]:
    name = f"batch_{batch:03d}"
    assert (backup / (name + ".pt")).samefile(formal / name / "_state/checkpoint_0016.pt")
patch = ["patch", "--batch", "--forward", "-p1", "-i", "scripts/wt103-c16-math.patch"]
subprocess.run(patch + ["--dry-run"], check=True)
subprocess.run(patch, check=True)
new = runtime()
assert new == dict(old, source=patched)
now = time.time()
record = {"first_patched_block_id":16, "run_id":root.name, "attention":"SDPBackend.MATH on all eight GPUs",
          "gradient_clip":"unchanged threshold; fail before Adam on nonfinite norm",
          "original_runtime":digest(old), "patched_runtime":digest(new),
          "original_deadline":progress["deadline_unix"], "failure_unix":status["unix_time"],
          "resumed_unix":now, "remaining_active_seconds":remaining, "new_deadline":now+remaining,
          "measurement_note":"Report block IDs 0-15 and 16 onward separately when comparing time or energy.",
          "patch_sha256":file_sha256("scripts/wt103-c16-math.patch"), "original_files":str(backup)}
atomic_json(root / "recovery_c16.json", record)
atomic_json(root / "native_environment.json", new)
atomic_json(root / "job.json", dict(job, native_runtime_sha256=digest(new)))
atomic_json(formal / "progress.json", dict(progress, deadline_unix=now+remaining))
print("Migrated block 16, remaining active hours", remaining / 3600)
