# Calibration protocol

Calibration is performed only on the validation split. Dataset ingestion must
split by stable speaker hash before any threshold calculation; the frozen test
split is used exactly once for the release report.

For each aligned phone, store the log-likelihood margin, duration, speaker,
expected phone and human error label. `calibrate_phone_thresholds` grid-searches
the cutoff subject to phoneme-error FAR ≤10% and correct-phone FRR ≤5%.
`calibrate_mismatch_margin` independently enforces wrong-word FAR ≤2%.

Results must be written to a new immutable calibration artifact before they are
activated, including dataset identifier, validation-split hash and generation
timestamp. The current `ScoringThresholds` defaults are engineering safeguards
and cannot be cited as measured performance.

Report metrics per phoneme and in aggregate, including sample and speaker
counts, confidence intervals, latency, model revision, lexicon version and
hardware. A calibration is invalid if a speaker appears in more than one split
or if any test label influenced threshold selection.
