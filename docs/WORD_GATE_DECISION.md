# Word identity gate decision

**Decision:** closed-vocabulary contrastive classification using the existing
temporal DTW benchmarks (`DtwWordIdentityGate`).

The exercise vocabulary is currently five known words. Comparing every
candidate is therefore cheaper, more inspectable, and easier to calibrate than
introducing open-vocabulary ASR. The gate converts negative DTW distances to a
closed-set probability distribution and requires a minimum winner margin.

`MATCH` allows phoneme evaluation to continue. A confident competing word
produces `MISMATCH`/`WRONG_WORD`; a small margin produces `UNCERTAIN`, never a
positive decision.

## Exit gates

- validate every one of the 20 ordered unequal word pairs;
- wrong-word false acceptance rate below 2%;
- p95 identity latency below 100 ms on target hardware;
- calibrate probability and margin on a speaker-disjoint validation set.

Until these gates are measured, the implementation is a versioned technical
baseline and not a validated child-speech classifier.

## Validation result

The speaker-disjoint validation split produced a calibrated margin of
`0.08588068716048952` with temporal factor-2 downsampling and wrong-word pair
FAR `1.953125%`, satisfying the isolated FAR budget. However, correct-word
false rejection is `81.25%`; the gate is therefore not MVP-ready. See
`config/word_gate_thresholds.v1.json` and
`docs/metrics/word_gate_validation.json`.
