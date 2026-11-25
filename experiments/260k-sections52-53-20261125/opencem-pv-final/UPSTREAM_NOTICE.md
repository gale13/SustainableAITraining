# Upstream provenance

This distribution calls the official `opencem.dataset.interpolate_value` and includes
the unmodified `dataset.py`, `clock.py`, `interfaces.py`, and `__init__.py` files from
https://github.com/OpenCEM-platform/opencem_simulator/tree/main/src/opencem .
Per-file Git object hashes and source URLs are recorded in `upstream_manifest.json`.
The retained subset supplies the official dataset APIs; it is not a claim to package
all components of the upstream simulator.

OpenCEM code and data remain attributable to their original authors. No new license
for upstream code is asserted here. The OpenCEM dataset metadata identifies the
dataset license as CC BY 4.0; preserve source attribution when sharing derived data.

Dataset source:
https://github.com/OpenCEM-platform/opencem-dataset/tree/5884d253a5267fb240b7a8df6fa9e4d49a905167/data

Citation supplied by the upstream repository:
T. S. Bartels, R. Wu, X. Lu, Y. Lu, F. Xia, H. Yang, Y. Chen, and T. Li.
"Bridging Natural Language and Microgrid Dynamics: A Context-Aware Simulator
and Dataset." arXiv:2604.05429, 2026.

Only the extraction/cleaning code, read-only daily API, quality catalog and associated
validation scripts in this directory were written for the present experiments.
