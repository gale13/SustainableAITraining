"""Validate equal exposure and IDs, then export only globally committed blocks."""
import argparse
import csv
import math
from pathlib import Path

from .common import atomic_json, read_json, write_csv


def audit(root):
    root = Path(root)
    run, progress = read_json(root / "run.json"), read_json(root / "progress.json")
    config, plan, manifest = run["config"], run["plan"], run["manifest"]
    if run["mode"] != "run":
        raise ValueError("Audit requires a formal run, not pilot results")
    count = progress["common_completed_blocks"]
    if count == 0:
        raise ValueError("No globally completed block yet")
    all_rows, summaries = [], []
    target = manifest["block_target_tokens"]
    for b in config["batches"]:
        previous_step = 0
        for block in range(count):
            directory = root / f"batch_{b:03d}" / f"block_{block:04d}"
            with (directory / "metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            expected_ids = [f"b{block:04d}_e{e:05d}" for e in range(1, plan["epochs_per_block"] + 1)]
            if [r["epoch_id"] for r in rows] != expected_ids:
                raise ValueError(f"Incomplete/duplicate epoch IDs in {directory}")
            for name in ("loss", "time", "energy"):
                with (directory / f"{name}.csv").open() as f:
                    if [r["epoch_id"] for r in csv.DictReader(f)] != expected_ids:
                        raise ValueError(f"Projection IDs differ: {directory}/{name}.csv")
            for i, row in enumerate(rows, 1):
                if int(row["batch_sequences"]) != b or int(row["block_id"]) != block or int(row["epoch_in_block"]) != i:
                    raise ValueError("Incorrect row identity")
                if int(row["train_tokens"]) != target:
                    raise ValueError("Unequal token exposure")
                steps = math.ceil(target / config["model"]["seq_len"] / b)
                previous_step += steps
                if int(row["optimizer_steps"]) != steps or int(row["global_steps"]) != previous_step:
                    raise ValueError("Optimizer-step continuity mismatch")
                if int(row["global_tokens"]) != target * (block * plan["epochs_per_block"] + i):
                    raise ValueError("Token continuity mismatch")
                for phase in ("train", "validation"):
                    for key in ("seconds", "energy_j", "mean_w"):
                        value = float(row[f"{phase}_{key}"])
                        if not math.isfinite(value) or value <= 0:
                            raise ValueError("Invalid time/energy")
                    if row[f"{phase}_energy_valid"] != "True":
                        raise ValueError("Invalid energy flag")
                    if not math.isclose(float(row[f"{phase}_energy_wh"]) * 3600, float(row[f"{phase}_energy_j"]), rel_tol=1e-8):
                        raise ValueError("J/Wh conversion mismatch")
                for key in ("train_loss", "validation_loss"):
                    if not math.isfinite(float(row[key])):
                        raise ValueError("Nonfinite loss")
            total_time = sum(float(r["train_seconds"]) for r in rows)
            total_j = sum(float(r["train_energy_j"]) for r in rows)
            summaries.append({"batch_sequences": b, "block_id": block, "epochs": len(rows),
                              "train_seconds": total_time, "train_energy_j": total_j,
                              "train_energy_wh": total_j / 3600,
                              "duration_target_met": total_time >= config["minimum_block_seconds"],
                              "validation_seconds": sum(float(r["validation_seconds"]) for r in rows),
                              "validation_energy_j": sum(float(r["validation_energy_j"]) for r in rows),
                              "last_validation_loss": float(rows[-1]["validation_loss"])})
            all_rows.extend(rows)
    write_csv(root / "all_epochs.csv", all_rows)
    write_csv(root / "block_summary.csv", summaries)
    met = all(r["duration_target_met"] for r in summaries)
    report = {"passed": met, "integrity_checks_passed": True,
              "common_completed_blocks": count, "planned_blocks": plan["planned_blocks"],
              "unused_full_blocks": len(manifest["blocks"]) - count, "batch_count": len(config["batches"]),
              "row_count": len(all_rows), "epochs_per_block": plan["epochs_per_block"],
              "all_duration_targets_met": met,
              "shortest_train_block_seconds": min(r["train_seconds"] for r in summaries),
              "scope": "GPU board energy in training/validation windows; waiting, warmup and checkpoints excluded",
              "uncommitted_outputs": "Any block >= common_completed_blocks and all attempts are excluded from analysis"}
    atomic_json(root / "audit.json", report)
    print(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_directory")
    audit(p.parse_args().run_directory)
