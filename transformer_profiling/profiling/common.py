import csv
import hashlib
import json
import math
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to write an empty results table")
    fields = fields or list(rows[0])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def export_block(directory, block_id, rows):
    """Identical primary keys in the combined table and all three projections."""
    directory = Path(directory) / f"block_{block_id:04d}"
    keys = ["epoch_id", "run_id", "batch_sequences", "block_id", "epoch_in_block", "global_epoch"]
    write_csv(directory / "metrics.csv", rows)
    for name, cols in {
        "loss": ["train_loss", "validation_loss", "validation_perplexity", "train_tokens", "unknown_fraction", "optimizer_steps", "global_steps", "learning_rate"],
        "time": ["train_seconds", "validation_seconds", "epoch_wall_seconds", "tokens_per_second"],
        "energy": ["train_energy_j", "train_energy_wh", "train_energy_method", "train_energy_valid", "train_mean_w", "train_power_integral_j", "train_counter_j", "train_counter_power_relative_difference", "train_sensor_note", "validation_energy_j", "validation_energy_wh", "validation_energy_method", "validation_energy_valid", "epoch_energy_j", "epoch_energy_wh"]
    }.items():
        write_csv(directory / f"{name}.csv", rows, keys + cols)


def make_plan(config, manifest, pilots, elapsed_seconds=0):
    batches = config["batches"]
    if sorted(p["batch_sequences"] for p in pilots) != sorted(batches):
        raise ValueError("Pilot batch list does not match the requested configuration")
    if any(p["peak_reserved_bytes"] > p["total_memory_bytes"] * config["memory_fraction_limit"] for p in pilots):
        raise ValueError("Insufficient measured VRAM headroom. Reduce the largest batches and rerun the whole pilot.")
    # Token exposure equivalent of the paper's 10,000 * 5,120, not 10,000 steps for every batch.
    paper_epochs = math.ceil(10000 * 5120 / manifest["block_target_tokens"])
    fastest = min(min(p["train_epoch_seconds"]) for p in pilots)
    slowest = max(max(t + v for t, v in zip(p["train_epoch_seconds"], p["validation_epoch_seconds"])) for p in pilots)
    if fastest <= 0 or not math.isfinite(slowest):
        raise ValueError("Invalid pilot timing")
    epochs = max(paper_epochs, math.ceil(config["minimum_block_seconds"] * config["minimum_duration_margin"] / fastest))
    available = config["budget_hours"] * 3600 - config["budget_reserve_seconds"] - elapsed_seconds
    block_seconds = epochs * slowest * config["runtime_margin"]
    blocks = min(len(manifest["blocks"]), math.floor(available / block_seconds))
    if blocks < 1:
        raise ValueError("No complete common block fits: relax the time requirement or narrow the batch range.")
    return {
        "config_sha256": digest(config), "manifest_sha256": digest(manifest),
        "epochs_per_block": epochs, "planned_blocks": blocks,
        "paper_equivalent_epochs": paper_epochs,
        "elapsed_before_plan_seconds": elapsed_seconds,
        "fastest_measured_train_epoch_seconds": fastest,
        "slowest_measured_train_plus_validation_epoch_seconds": slowest,
        "predicted_fastest_train_block_seconds": epochs * fastest,
        "predicted_slowest_block_seconds_with_margin": block_seconds,
        "predicted_total_hours_with_margin": block_seconds * blocks / 3600,
        "budget_hours": config["budget_hours"],
        "interpretation": "Calibrated estimate, not a hard duration guarantee; no artificial sleep counted as training."
    }
