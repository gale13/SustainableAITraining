# Current Section 5.3 V figure — QA

The requested source is the prior V-metrics figure's first two panels. Structural adaptation preserves the meaning of the horizontal variable V, the three sigma_e groups, TARS in W, and epoch-start Q in seconds. Numerical values are newly simulated with the requested defaults; old fixed-weight curves are not reused.

## Data and scope

- 33 sigma_e/V combinations × 240 dates × 32 paired noise seeds = 253,440 new runs, each with 120 epochs. No candidate point or date was dropped after seeing its result.
- Each panel contains three curves of 11 parameter points. Each point first averages seeds within date, then gives all dates equal weight. Per-date values and all plotted intervals are supplied as CSV.
- Pointwise 95% intervals use 2,000 paired calendar-week cluster bootstrap resamples. No hypothesis-test or multiplicity-adjusted superiority claim is attached to the curves.
- Target violations are disclosed in the caption and full table: at sigma_e=.23 and V=.1,1,10,100, counts are 9,3213,3625,3671 out of 7680. The points remain visible. There are no observed violations at the other settings.
- The user explicitly requested only panels a and b. Other quantities remain in machine-readable tables; no extra V/frontier panels or figures were produced. Historical outputs are excluded from the current manuscript figure set.
- The frozen model, empirical environment and all 240 dates remain unchanged. The 45 dates relying on >5-minute interpolation gaps and stale held observations remain documented limitations, with numerical quality-subset comparisons supplied.

## Numerical verification

`verification.json` reports PASS. Checks include official PV energy windows and past-only lookup across 240 dates; 750 MLP states; 4,320 independent Python replay epochs covering all three sigma_e values; fixed-state counterfactual changes to noise and future PV; and 99 standalone decisions with only the two model JSON files present. The Section 5.2 regression matches all 7,680 common runs exactly on checked outcome metrics. All runs finish within 24 hours and obey the finite-window queue bound up to floating-point roundoff.

## Rendered review and exports

The actual PDF proof was visually inspected: only panels (a) and (b), a shared three-item legend, readable labels, grayscale marker distinctions, visible uncertainty bands, and no clipped or overlapping annotations. All 33 parameter points and their intervals fit inside each panel. The log transforms have explicit positive V and positive lower queue-interval guards; no pseudocount was added.

Python/matplotlib generated all outputs. Final dimensions are 6.5 × 3.0 inches. Body/tick text is 9–10 pt Times New Roman; math uses STIX and the smallest 6.3 pt text is a mathematical exponent. All PDF fonts are embedded and text remains selectable. SVG text is editable. PNG is a 300 dpi preview; TIFF is 3900 × 1800 pixels at 600 dpi. The automated PDF check found no text bounding boxes outside the page.

## Generic source-checker findings

The unchanged generic preflight output is retained as `results/figure_source_preflight.json`; it is not represented as an unqualified PASS. Its findings are resolved as follows:

- Font-family FAIL: the source checker expects a sans-serif family. Times New Roman is explicitly required by the established OR figure contract; exported PDF fonts verify the intended family. This is a target-specific override.
- Raster-DPI FAIL: the regular expression matches the suffix `res=1` inside `figures=1` in numerical figure metadata. That field counts figures and is not a resolution. Actual save calls specify 300/600 dpi, and TIFF dimensions and dpi were independently verified.
- Width WARN: 165.1 mm is the requested 6.5-inch OR figure width, rather than the generic Nature 89/183 mm defaults.
- Log-guard WARN: positivity is explicitly asserted with `data.V.gt(0)` and `data.Q_ci_low.gt(0)`, which this source heuristic does not recognize. All plotted lower bounds are positive.

The resolved findings do not require changing scientific data or the user's manuscript style. Current journal submission requirements should be checked at submission time.
