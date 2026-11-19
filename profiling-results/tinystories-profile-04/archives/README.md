# Complete server archive

Download `profiling04_full_server.tar.gz` from the [Profiling 04 release](https://github.com/gale13/SustainableAITraining/releases/tag/profiling04_2026_11_20).

It contains all 150 final model/optimizer/RNG checkpoints, all raw power traces, per-task metrics and metadata, worker logs, launch/recovery tests and deployed code. The several-GB public corpus and regenerable compiler caches are excluded. Dataset source hashes and preparation scripts are included in the repository.

Verify with `shasum -a 256 profiling04_full_server.tar.gz` against `release_assets.json`, then extract with `tar -xzf profiling04_full_server.tar.gz` into an empty directory. `server_file_manifest.json` lists each public file's SHA-256 and original server hash. Numeric measurements and checkpoint bytes are unchanged; user paths and hardware UUIDs in text metadata are replaced with portable identifiers.
