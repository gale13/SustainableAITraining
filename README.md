# WikiText-103 Transformer profiling

Version **profiling_test_11_13** archives the completed `wt103-01` experiment started on 13 November 2026. Eight RTX 5090 GPUs each ran a separate persistent, fixed-batch trajectory using batch sizes 4, 8, 10, 16, 24, 32, 48 and 64. All eight completed 19 common data blocks at 229 epochs per block, yielding 34,808 formal epoch records. Block identifiers in the CSVs start at zero. The pilot and interrupted attempts are retained separately from the formal epoch table.

## Data and results

| File or directory | Contents |
| --- | --- |
| [Formal epoch table](profiling-results/wt103-01/training/all_epochs.csv) | Training and validation loss, duration, energy, token counts and sensor diagnostics |
| [Block summary](profiling-results/wt103-01/training/block_summary.csv) · [Integrity audit](profiling-results/wt103-01/training/audit.json) | Per-block summaries and completion checks |
| [GPU runs](profiling-results/wt103-01/training/) · [Pilot](profiling-results/wt103-01/pilot/) | Per-block loss, time and energy CSVs, timestamped power telemetry, environment records, worker logs and attempt histories |
| [Absolute duration and energy](profiling-results/wt103-01/figures/absolute/) · [Validation-loss figures](profiling-results/wt103-01/figures/raw_loss/) | Task-level measurements and original validation-loss trajectories |
| [229-epoch conditional means and action profiles](profiling-results/wt103-01/figures/conditional_means_v1/action_profiles_229.pdf) · [Fit coefficients and scripts](profiling-results/wt103-01/figures/conditional_means_v1/) | Full-horizon fixed-state fits, action-wise averages, shorter-horizon analysis and held-out diagnostics |
| [Source and native setup](transformer_profiling/README_NATIVE.md) · [Docker setup](transformer_profiling/README.md) | Server-side training code, environment setup, configuration and tests |
| [Recovery record](profiling-results/wt103-01/recovery_c16.json) · [Pre-patch code and diagnostics](recovery/block_0016/) | Runtime change and restart at the boundary after 16 completed blocks |
| [File manifest](manifest.csv) | Relative path, byte count and SHA-256 of each published experiment file |

The first 16 blocks used the original attention implementation. The final three used the math attention backend after recovery from a numerical failure. Compare time and energy within the same backend. The full-horizon conditional figure uses blocks 17–19 (one-based), with 5,496 epoch measurements and 5,472 observable pre/post loss transitions. The measured energy is GPU-board energy. The fits use training duration in seconds and the bilinear loss metric defined as duration times the normalized running-minimum training loss after the epoch. Validation-loss plots use the recorded validation data.

## Reproduce the figures

From the repository root, enter the [run directory](profiling-results/wt103-01/) in a Python environment with NumPy and Matplotlib. The commands use the checked-in measurements:

```sh
cd profiling-results/wt103-01
python figures/conditional_means_v1/fit_profiles.py
python figures/conditional_means_v1/plot_actions_229.py
```

The [native runtime record](profiling-results/wt103-01/native_environment.json) includes package versions and training-source hashes. The [data manifest](profiling-results/wt103-01/training/data_manifest.json) describes the prepared corpus. The public snapshot preserves numeric measurements and replaces hardware UUIDs and user home paths with portable identifiers. Model checkpoints, dataset binaries and installed environments are excluded.
