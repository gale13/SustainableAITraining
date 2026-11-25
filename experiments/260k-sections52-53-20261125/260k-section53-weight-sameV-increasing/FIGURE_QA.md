# Figure QA — same-V comparison against Increasing

Checked 2026-11-25. **PASS with documented source-checker overrides.**

## Data and interpretation

All 240 requested dates, 32 paired seeds, 17 V values and five schedules are included: 652,800 comparison runs, 78,336,000 simulated epochs, 85 absolute metrics and 68 paired effects. This includes 122,880 new Increasing runs, 7,680 prior Increasing runs and 522,240 reused decay runs. Reuse is explicit, with source hashes and matching configuration checks. There is no outcome-based exclusion.

The denominator is mean Increasing TARS at the same V, always positive. All negative/zero values are retained. The largest common benefit is 24.59%–24.94% at V=.001; at V=.1 the gain is negative (about −.0605%, pointwise interval below zero). These tiny negative values are necessarily close to the zero line at the full-scale plotting resolution; exact values and intervals are reported in README and CSV. No claim of uniform or separately tuned superiority is made.

Increasing=(100,1000,10000) is the pre-existing comparator, not a newly selected weak baseline and not the increasing-batch INC policy. Four decay families retain the original functions and matched initial weight. Weights are not normalized to equal sum. This study concerns the explicit schedules, not all increasing functions.

Each date receives equal weight after averaging 32 noise realizations. Shading shows 2,000 paired calendar-week bootstrap pointwise 95% intervals, without a simultaneous-coverage claim. No new MLP fitting, train/test split, date choice or PV scaling occurs. The curves are model-based simulation results, not new hardware observations.

All comparison runs finish within a day and have no observed DWAE>.235; the maximum observed DWAE is .2335692026016962. An additional 1,800 epochs were independently replayed for Increasing, including mean-only decisions without noise files, past observations, model means, actions, actual noise, official PV energy and queues. Saved metrics match fresh replay to numerical precision. The existing decay validation applies to the reused immutable runs; 7,680 Increasing records match the historical reference with zero compared-metric error.

## Visual and export inspection

Inspected both the actual PNG and PDF proof `qa/sameV_increasing_pdf_proof.png`. Title, legend, axis labels, endpoints, marker styles, zero line and uncertainty bands are readable; no clipped or overlapping text was observed. Four curves share the original grayscale styles. This is a structural adaptation of the Fixed companion, changing only comparator data/labels; no fitted curves or manual point edits.

One panel, 3.5×3.2 inches (88.9 mm wide), main Times New Roman text 9–10 pt. Math uses STIX; the smallest extracted glyph is a 6.3 pt subscript. PDF fonts are embedded and all text bounding boxes remain inside the page. SVG has editable text. TIFF is 2100×1920 pixels at 600 dpi; PNG is 1050×960 pixels. Full V range and all 68 effects are included, including negative intervals. The original Fixed plot and Section 5.3 V panels a,b remain unchanged.

## Generic source checker overrides

`results/figure_source_preflight.json` is the unmodified report: 12 PASS, one FAIL, one WARN.

- FONT-FAMILY FAIL: generic Nature-oriented regex does not recognize the explicit Times New Roman serif setting. This intentionally follows the requested OR style. Actual PDF extraction verifies embedded TimesNewRomanPSMT and STIX fonts; retain them.
- LOG-GUARD WARN: source explicitly checks `c.V.gt(0).all()` and verifies the full grid; all V are positive. The generic regex misses this pandas guard. Increasing denominators are also explicitly checked positive. No pseudocount or omitted row is used.

No-demo-generator PASS applies only to the plotting source. This is deliberately a simulation study, identified as such throughout the report.

`verification.json` records numerical and export checks. `checksums.json` covers scripts, results, reports and figures; `protocol.json` hashes the immutable source experiments and frozen simulator. Raw Increasing runs remain in daily checkpoints and the aggregate CSV; decay raw runs are referenced from the preceding experiment without rerunning or duplicating them.
