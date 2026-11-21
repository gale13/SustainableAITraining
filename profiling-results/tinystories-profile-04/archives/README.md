# Complete server archive

Download `profiling04_full_server.tar.gz` from the [Profiling 04 release](https://github.com/gale13/SustainableAITraining/releases/tag/profiling04_2026_11_20).

It contains all 150 final model/optimizer/RNG checkpoints, all raw power traces, per-task metrics and metadata, worker logs, launch/recovery tests and deployed code. The several-GB public corpus and regenerable compiler caches are excluded. Dataset source hashes and preparation scripts are included in the repository.

Verify with `shasum -a 256 profiling04_full_server.tar.gz` against `release_assets.json`, then extract with `tar -xzf profiling04_full_server.tar.gz` into an empty directory. `server_file_manifest.json` lists each public file's SHA-256 and original server hash. Numeric measurements and checkpoint bytes are unchanged; user paths and hardware UUIDs in text metadata are replaced with portable identifiers.


## Simulator replacement (2026-11-21)

The same release's `profiling04_simulator.zip` now contains the online train_loss
simulator, its refitted models and complete noise diagnostics. Download the new
`SHA256SUMS` with it. The full server archive is unchanged. The release tag and
GitHub-generated source archives identify the original training snapshot;
the current simulator source is in `main`, and the replacement ZIP corresponds
to the update commit linked in the release notes.

The current simulator defaults to validation-only empirical residual resampling,
with a fresh joint draw each epoch. See `simulator/EMPIRICAL_NOISE.md`.


## Separate full-data simulator

`profiling04_simulator_full_data.zip` contains the all-600-observation refit,
with its own `profiling04_simulator_full_data.zip.sha256` checksum asset.
`profiling04_simulator.zip` and its existing checksum remain the preserved
feasibility experiment. Neither that ZIP nor the full training archive is
replaced. See `../SIMULATORS.md` for their different evaluation roles.
