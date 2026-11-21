# Profiling 04 — TinyStories-260K

Complete experiment and simulator archive: 5 batch sizes × 30 paired seeds × 4 epochs = **600 epoch observations**, all 150 tasks completed.

| GPU index | Batch size | Tasks | Epochs per task | Mean task wall time |
|---|---:|---:|---:|---|
| 0 | 32 | 30 | 4 | 42m 41s |
| 1 | 64 | 30 | 4 | 21m 10s |
| 2 | 128 | 30 | 4 | 12m 30s |
| 3 | 256 | 30 | 4 | 12m 24s |
| 4 | 512 | 30 | 4 | 13m 31s |

Task seeds are 1337–1366, paired across batches. Each task resets model weights, optimizer and random state. The published stories260K Llama-style configuration has 260,032 parameters, vocabulary 512 and context length 512. The official 100,000 updates at batch 128 equal 3.613969 passes over this corpus; this experiment uses 4 full epochs, without gradient accumulation. Warmup and cosine decay are indexed by processed data, with decay ending at epoch 4.

## Contents

- [Training code and recipe](code/stories260k/README.md): exact architecture, packing, training, energy sampling, checkpoint recovery and fixed-GPU dispatcher; upstream source and license included.
- [Original epoch measurements](experiment/epochs.csv): all 600 rows, original train/validation/probe losses, timing and energy counters.
- [Initial measurements](experiment/baselines.json), [initialization hashes](experiment/initials.json), [completion records](experiment/completed.json), [dataset manifest](experiment/manifest.json).
- [Three fitted MLPs, simulator and analysis](simulator/README.md): complete split assignments, weights, optimizer outcomes, MSE/RMSE, residual diagnostics, figures and validation-only noise library.
- [Launch and progress provenance](provenance/): initial-weight checks, smoke-test recovery, deployment hashes and timestamped snapshots.
- `archives/`: complete server-record archive, including raw 0.5-second power traces, worker logs and all 150 final model/optimizer/RNG checkpoints. See its manifest for reconstruction.
- `manifest.csv`: SHA-256 and byte size of every file in this publication.

## Full-data simulator for simulation use

Two distinct versions are retained; see [the version guide](SIMULATORS.md).

- [Full-data simulator](simulator-full-data/README.md): all **600 observations**
  refit the three MLPs and supply the empirical residual bank. Each action/epoch
  has **30 joint residual donors**. Architecture, regularization and phase
  thresholds are fixed from the feasibility experiment; there is no held-out
  set in this final calibration. Calibration RMSE is 1.31946 s for duration,
  723.988 J for GPU demand, and 0.359819 s for time-weighted loss.
- [Preserved feasibility experiment](simulator/README.md): 360 training rows,
  120 validation residuals and 120 independent test rows. Its files and existing
  release ZIP are unchanged; use this version as the independent evaluation.

The full-data directory includes source, fitted weights, all residual donors,
noise summaries, figures, sample rollouts, verification and SHA-256 checksums.
Its separate release asset is `profiling04_simulator_full_data.zip`.

## Feasibility simulator result — updated 2026-11-21

The default simulator now uses the running minimum of **completed online epoch
train_loss**, with the measured pretraining probe retained only as the initial
normalization anchor. The pre-epoch input is `(J, phase, action)`; the current
epoch outcome is never included in its own input. K=3, W=1, and training-only
thresholds are -0.3401582057 and -0.0505373623.

The paired-seed split remains 18 train / 6 validation / 6 test (360/120/120 epoch
rows). All three selected MLPs converged. Test RMSE is **1.62203 s** for duration,
**709.323 J** for GPU demand, and **0.39290 s** for T*J_next. The last target changes
with the loss instrument, so its raw RMSE is not a same-label comparison with
the old probe model. See [the fit report](simulator/README.md).

[Updated noise analysis](simulator/NOISE_REPORT.md) uses all 120 held-out points.
Empirical residual SD is 1.61658 s, 702.928 J and 0.389383 s respectively; these
are not validated shared Gaussian sigmas. Action-dependent scales, skew and
tails remain. The default noise is [empirical validation-residual resampling](simulator/EMPIRICAL_NOISE.md):
each epoch draws a fresh joint triple from the matching action/epoch, with mean
calibration and physical bounds. No Gaussian noise family is imposed. Test data
are used only for evaluation. Joint four-epoch trajectories remain an explicit
sensitivity option. Exact task-total comparisons
enumerate 1,296 independent paths and six joint paths per batch. Joint draws
better match energy on average, while independent draws better match duration
on average in these six held-out seeds; neither universally wins.

The replaced probe version remains available in Git history. `comparison_reference/`
contains only numeric comparison data and thresholds, with no old model weights. The original
language-model training code, checkpoints and 600 measurements are unchanged.

## Reproduce analysis

```bash
cd profiling-results/tinystories-profile-04/simulator
python -m pip install -r requirements.txt
python fit.py
python diagnostics.py
python plots.py
python noise_analysis.py
python verify.py
python compare.py
```

Inference requires NumPy only. Running the original experiment additionally requires compatible CUDA/PyTorch, five available GPUs, SentencePiece and NVML. Copy `code/` into a chosen experiment root's `code/`, use the root argument for `python -m stories260k.prepare ROOT`, then `python -m stories260k.launch ROOT`. Run from `ROOT/code`. The archived `start.sh` records the original launch approach with generic paths; configure these paths for your host. `stories260k.status ROOT` prints progress.

## Scope and integrity

Public numeric measurements, random seeds, weights and split membership are preserved. User-specific home paths and physical GPU UUIDs are replaced with generic paths and stable device-00..04 identifiers, following the repository's first profiling archive convention. Original-source hashes remain provenance values; the publication manifest hashes the actual public files.

The several-GB public corpus, installed Conda environment and regenerable Triton/Inductor caches are not copied into Git. Corpus download URLs, exact SHA-256, tokenizer, packing code and dependency versions are included. No credentials or unrelated projects are part of this publication.
