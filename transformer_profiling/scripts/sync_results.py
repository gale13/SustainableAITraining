"""Host-side, explicitly invoked results export/commit/push. Never upload model states."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True).stdout.strip()


def pack(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    allowed = {".csv", ".json", ".txt", ".log", ".gz"}
    with tempfile.TemporaryDirectory(prefix="profiling-export-") as temp:
        archive = Path(temp) / "results.tar.gz"
        included = []
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source)
                if not path.is_file() or path.is_symlink() or "_state" in relative.parts:
                    continue
                if path.suffix not in allowed:
                    continue
                tar.add(path, arcname=str(Path(source.name) / relative), recursive=False)
                included.append(str(relative))
        whole_sha = hashlib.sha256()
        parts = []
        with archive.open("rb") as f:
            index = 0
            while data := f.read(40 * 1024 * 1024):
                name = f"results.tar.gz.part{index:04d}"
                (destination / name).write_bytes(data)
                whole_sha.update(data)
                parts.append({"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                index += 1
        (destination / "manifest.json").write_text(json.dumps({
            "archive_sha256": whole_sha.hexdigest(), "parts": parts, "included_files": included,
            "excluded": ["model and optimizer checkpoints", "dataset", "image", "credentials"],
            "restore": "cat results.tar.gz.part* > results.tar.gz && tar -xzf results.tar.gz"
        }, indent=2) + "\n")
        (destination / "SHA256SUMS").write_text("".join(
            f"{part['sha256']}  {part['file']}\n" for part in parts))
        (destination / "archive.sha256").write_text(f"{whole_sha.hexdigest()}  results.tar.gz\n")
    # Expose the audited tables directly in GitHub as well as in the full archive.
    formal = source / "training" if (source / "job.json").exists() else source
    if (formal / "audit.json").exists():
        committed = json.loads((formal / "audit.json").read_text())["common_completed_blocks"]
        tables = [formal / name for name in ("all_epochs.csv", "block_summary.csv", "audit.json")]
        for path in formal.glob("batch_*/block_*/*.csv"):
            if int(path.parent.name.split("_")[-1]) < committed:
                tables.append(path)
        for path in tables:
            target = destination / "tables" / path.relative_to(formal)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def wait_for_completion(source):
    while True:
        status_file = source / "status.json"
        if status_file.is_file():
            status = json.loads(status_file.read_text())
            if status["stage"] in ("complete", "complete_with_duration_shortfall"):
                return
            if status["stage"] == "failed":
                raise RuntimeError(f"Profiling failed: {status.get('error')}. See {status_file}.")
        time.sleep(20)


def pack_once(source, destination, identity):
    marker = destination / "source.sha256"
    if destination.exists():
        if not marker.is_file() or marker.read_text().strip() != identity:
            raise RuntimeError(f"Another export owns {destination}; choose a distinct run ID or destination")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "export"
        pack(source, staged)
        (staged / "source.sha256").write_text(identity + "\n")
        staged.rename(destination)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="A stopped run or completed pilot output directory")
    p.add_argument("--export-dir", help="Pack locally without git, e.g. /backup/run1")
    p.add_argument("--repo", help="Local clone of gale13/SustainableAITraining")
    p.add_argument("--push", action="store_true", help="Push the created commit to origin")
    p.add_argument("--wait", action="store_true", help="Wait for this pipeline run to finish and pass its integrity audit")
    args = p.parse_args()
    source = Path(args.run).resolve()
    if bool(args.repo) == bool(args.export_dir):
        p.error("Choose exactly one of --repo and --export-dir")
    if args.push and not args.repo:
        p.error("--push requires --repo")
    if args.wait:
        wait_for_completion(source)
    metadata = source / "job.json"
    if not metadata.exists():
        metadata = source / "run.json"
    if not metadata.exists():
        p.error("Not a profiling output directory")
    # Lock cooperates with launcher to prevent exporting live, inconsistent artifacts.
    import fcntl
    with (source / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if args.wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        if metadata.name == "job.json":
            state = json.loads((source / "status.json").read_text())["stage"]
            if args.repo and state not in ("complete", "complete_with_duration_shortfall"):
                p.error("The job is unfinished. Use --wait to publish on completion, or --export-dir for a snapshot.")
        identity = hashlib.sha256(metadata.read_bytes()).hexdigest()
        if args.export_dir:
            destination = Path(args.export_dir).resolve()
            if destination == source or source in destination.parents:
                p.error("Export destination must be outside the run directory")
            if destination.exists():
                p.error("Choose a new export directory for this snapshot")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
                staged = Path(temp) / "export"
                pack(source, staged)
                staged.rename(destination)
            print(destination)
            return
        repo = Path(args.repo).resolve()
        origin = git(repo, "remote", "get-url", "--push", "origin")
        if origin.rstrip("/").removesuffix(".git") not in (
            "https://github.com/gale13/SustainableAITraining", "git@github.com:gale13/SustainableAITraining"
        ):
            p.error("origin is not gale13/SustainableAITraining; refusing to upload")
        branch = git(repo, "branch", "--show-current")
        if not branch:
            p.error("Check out a branch in the destination repository")
        relative = Path("profiling-results") / source.name
        pack_once(source, repo / relative, identity)
        git(repo, "add", "--", str(relative))
        if git(repo, "diff", "--cached", "--name-only", "--", str(relative)):
            git(repo, "commit", "--only", "-m", f"Archive Transformer profiling results: {source.name}", "--", str(relative))
        if args.push:
            git(repo, "push", "-u", "origin", branch)
        print(f"Saved {relative}; commit {git(repo, 'rev-parse', '--short', 'HEAD')}; pushed={args.push}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr.strip() or str(exc)) from exc
