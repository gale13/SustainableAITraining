# Combined figure QA

PASS with documented generic source-checker overrides, 2026-11-25.

- Structural layout adaptation from two verified quantitative panels; no new fitting or simulation. All 68 source rows per panel retained. Copied source CSVs are byte-identical; plotted y arrays are checked against source values. Source hashes unchanged after plotting. Both panels use identical decay means and matching protocols.
- Same positive log-V grid and common y limits include every point and confidence interval, including negative Increasing comparisons. No data exclusions, smoothing, rounding or manual graphical edits.
- Inspected the actual exported PDF raster proof at 200 dpi. Both panels (a,b), shared four-item legend, complete axis labels, ticks and intervals are readable with no observed overlap or clipping.
- One-page PDF, 7.2×3.2 inches, embedded Times New Roman/STIX, all extracted text within page bounds. SVG editable text. TIFF 4320×1920 pixels at 600 dpi; PNG 2160×960 pixels. Main text 9–10 pt.
- The unmodified generic checker report is qa/source_preflight.json: 11 PASS, two FAIL, one WARN. FONT-FAMILY is overridden for the established OR Times New Roman style, verified in the actual PDF. RASTER-DPI detects the 200 dpi PDF inspection proof; that file is only a QA preview. The delivered TIFF is explicitly verified as 600 dpi and the delivered PNG is 300 dpi, so no production raster is below 300 dpi. LOG-GUARD is resolved by explicit positive-V and positive-denominator checks in the source. No pseudocounts or discarded points are used. The export width is 182.88 mm, a full-width figure, and the width check passes.
- These are calibrated simulation outputs with paired dates/seeds, not new hardware measurements or two independent repeated experiments. Original per-panel statistics, pointwise interval interpretation and V-dependent claim boundaries remain unchanged.
