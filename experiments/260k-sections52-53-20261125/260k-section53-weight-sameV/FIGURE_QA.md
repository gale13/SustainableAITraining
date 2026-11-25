# Figure QA — same-V weight-decay comparison

Checked on 2026-11-25. Status: **PASS with documented source-checker overrides**.

## Data and interpretation

- All 240 requested PV dates, all 32 paired seeds, all five schedules and all 17 predeclared V settings are retained: 652,800 runs, 78,336,000 simulated epochs. No date or V filtering by outcome.
- `source_data/sameV_curves.csv` contains all 68 effects. `source_data/absolute_metrics.csv` contains all 85 settings, including Fixed, absolute TARS, DWAE and feasibility.
- The ordinate is the percentage difference of equally weighted date means, relative to Fixed at exactly the same V. Lines connect evaluated settings without fitting or smoothing. Light bands show pointwise paired calendar-week bootstrap 95% intervals, not simultaneous bands.
- All 652,800 runs have observed DWAE below .235; maximum observed DWAE is .2335414697436942. All finish within one day; maximum duration is 21.294761583681208 hours.
- Independent Python replay verified 9,000 epochs, including all five schedules and five V settings. The controller was given only model JSONs, not noise files; actual noise and official PV energy were checked separately. The unchanged core inherits the verified future-PV/noise-isolation checks. All 115,200 shared runs match the preceding two experiments exactly in compared metrics.
- Benefits are localized to small V. All four families improve mean TARS by 17.94%–19.33% at the common V=3e-4; large-V plateaus and extreme-small-V zeros remain visible. Each schedule's separately best grid value is .1, with a negligible, statistically unresolved decay advantage. The figure does not establish uniform strict dominance or superiority after separate V tuning.
- This is an exploratory follow-up using the user's screenshot sigma=.235, not a new blind confirmation. Full model, interpolation and simulation limitations are recorded in README.md.

## Visual inspection and export

- Inspected the actual exported PNG and the PDF raster proof `qa/sameV_pdf_proof.png`, including title, legend, ticks, zero line, confidence bands and endpoints. No clipping, overlapping labels or legend/data conflict was observed.
- One compact grayscale panel, 3.5 × 3.2 inches (88.9 mm wide), with four distinct marker/line styles. Main text is Times New Roman, 9–10 pt; math uses STIX. The smallest extracted PDF glyph is a 6.3 pt mathematical subscript.
- PDF is one page with embedded Times New Roman and STIX fonts. All extracted text bounding boxes lie within the page. SVG retains editable text. TIFF is 2100 × 1920 pixels at 600 dpi; PNG is 1050 × 960 pixels.
- Only the four exports of `weight_decay_sameV` are in the current figure directory. Existing Section 5.3 V panels a,b and frozen historical experiments are unchanged.

## Source checker findings

The unmodified generic checker report is `results/figure_source_preflight.json`: 12 PASS, one FAIL, one WARN. It is supplemented by actual export inspection, not reported as a clean generic pass.

- **FONT-FAMILY FAIL — override:** the generic Nature-oriented source checker does not recognize the explicit Times New Roman configuration as its expected publication font. The requested OR style uses Times New Roman; actual PDF extraction verifies `TimesNewRomanPSMT`, `STIXGeneral-Regular`, and `STIXGeneral-Italic`, all embedded. Retain these fonts.
- **LOG-GUARD WARN — override:** the plot source asserts `c.V.gt(0).all()` and validates all 17 V settings. Every value is strictly positive (1e-6 through 100); no rows are omitted and no pseudocount is used. The source regex does not recognize this pandas guard.

The checker’s no-demo-generator finding describes the plotting source only. The data are explicitly simulator results, not new hardware measurements or independent empirical observations.

## Reproducibility

`verification.json` records the completed numerical/export checks. `protocol.json`, `cache_manifest.json`, `checksums.json`, and `environment.json` preserve configuration and file provenance. A one-ULP CSV round-trip cache-validator repair is documented in `cache_validation_fix.json`; it did not change any simulated input or result. Rerunning `finalize.py` validates existing outputs and refreshes checksums without rerunning the experiment.
