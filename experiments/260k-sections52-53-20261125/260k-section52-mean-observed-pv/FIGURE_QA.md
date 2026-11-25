# Figure 8 verification — 2026-11-25

## Evidence and scope

The quantitative composite shows all 240 daily PV-generation totals, the seven policies' mean renewable shortfall, and their daily DWAE–TARS outcomes. Under the specified mean-only, last-observation controller, LYA has the lowest average TARS among the six baselines and LYA at PV70. This is a calibrated simulation result, not a new physical GPU experiment or a guarantee of superiority on every day.

- All 240 dates are retained, including the 45 dates with environmental interpolation intervals longer than five minutes.
- Each policy/date mean uses all 32 paired noise realizations, with 30 tasks and four epochs per task. The main figure uses 53,760 runs.
- The heatmap contains 240 cells and the scatter contains all 1,680 policy/date points. No points were sampled, trimmed, or removed to improve layout.
- Daily comparison with the best feasible baseline gives 230 wins, nine losses, and one tie for LYA. These are comparisons of daily means, not daily significance tests.
- Error bars are 95% paired calendar-week bootstrap intervals using 2,000 resamples. The frozen MLP fit uncertainty and unknown weather inside long gaps are not included.
- V was chosen only on the original 48 tuning dates using eight noise seeds per date. The 192 remaining dates are reported separately. No model retraining or baseline tuning was performed.
- B512 and DEC exceed the DWAE constraint and remain visible, with hatching. LYA has no observed violations in its 7,680 PV70 runs; this does not imply a zero population violation probability.

## Controller and environment checks

`verification.json`, `results/engine_verification.json`, and `results/standalone_controller_verification.json` report PASS. The standalone controller runs with only the two model JSON files. Fixed state and observed power produce identical scores and actions when future PV or the environment's noise bank changes. Actual environment outcomes remain noise-sensitive.

The audit replays one paired seed on all 240 dates, totaling 28,800 epochs. Every recorded observation timestamp is at or before its decision time, estimated PV equals last observed power times MLP mean duration, and predicted shortfall is its positive-part difference from mean demand. Environmental PV still uses the realized duration and the validated official piecewise-linear energy integral.

Source experiment hashes and frozen simulator hashes are unchanged. The six baseline records and historical LYA control retain their prior outcomes. New LYA records are freshly simulated and are not relabeled old results.

## Data limitations retained in the figure

Twelve midnight initializations have an observation older than 300 seconds. The oldest initial observation is 25.4595 days old; 5.1090% of PV70 decisions use a sample older than 300 seconds. These gaps are documented in the report and age tables. No interpolated or future sample was substituted for a real historical observation. The 195-, 213-, and 142-day quality subsets are supplied separately.

The lowest daily DWAE is approximately 0.223426. The scatter's lower x bound was expanded to include it with padding; the former 0.224 lower bound was not used to discard this point. All plotted coordinates and intervals are within their axes. PV heatmap energy uses the same PV70 scaling as the simulation.

## Rendered export review

The rendered main PDF proof and all six supplementary figures were visually inspected. Labels, direct annotations, axes, hatching, and legends are readable and do not overlap materially or extend beyond the page. All figures use the paper's terms LYA, time-averaged renewable shortfall (TARS), and duration-weighted average error (DWAE).

- Python/matplotlib is the sole plotting backend; Python/PyMuPDF rendered the PDF proof.
- Main figure: 6.5 × 5.8 inches, Times New Roman, minimum font size 9 pt, grayscale.
- All seven PDFs: 6.5-inch width, embedded Times New Roman, selectable text, and no text bounding box outside the page.
- PDF/SVG are the vector deliverables. PNGs are previews. The main TIFF is 3900 × 3480 pixels at 600 dpi.
- Source data, figure configuration, scripts, captions, and reproduction instructions are supplied alongside the exports.

## Source-preflight findings and explicit resolutions

The generic Nature-oriented source checker was run on both plotting scripts; its original outputs are preserved under `results/`. Its raw reports are not described as an unqualified PASS.

- **Font-family FAIL, both scripts:** resolved by the project's OR figure specification. The generic check recognizes a sans-serif default, whereas the requested manuscript style uses Times New Roman. Actual PDF extraction confirms consistent embedded Times New Roman and minimum 9 pt text. Changing the font merely to satisfy the generic heuristic would violate the local figure contract.
- **Width WARN:** 165.1 mm (6.5 inches) is the established OR manuscript figure width, rather than the checker's 89/183 mm Nature defaults.
- **Supplement raster WARNs:** supplementary PDF/SVG are the intended vector deliverables; their 300 dpi PNGs are previews. The main figure additionally includes the required 600 dpi TIFF.
- **Supplement log-guard WARN:** the logarithmic axis uses the explicit positive capacity grid 1, 2, 4, 8, and 16. No zero or negative multiplier is plotted or dropped.

These findings are resolved against the local figure contract and rendered-output evidence, without altering or concealing the generic checker reports. Final journal submission specifications should be checked when submitting.
