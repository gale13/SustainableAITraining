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

## Simulator result — updated 2026-11-21

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
tails remain. The noise bank uses validation seeds only and supports independent
epoch draws and joint four-epoch trajectories. Exact task-total comparisons
enumerate 1,296 independent paths and six joint paths per batch. Joint draws
better match energy on average, while independent draws better match duration
on average in these six held-out seeds; neither universally wins.

The replaced probe version remains available in Git history. `previous_probe/`
contains explicit comparison snapshots, not the default model. The original
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
