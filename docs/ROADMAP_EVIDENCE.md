# Roadmap completion evidence

Updated: 2026-07-24

Every checked item in `ROADMAP.md` is backed by source, tests, a generated
contract, or a measured report. The remaining unchecked items are release
outcomes rather than unimplemented local scaffolding.

| Remaining gate | Current evidence | Required external change |
| --- | --- | --- |
| Phoneme model load/inference/RSS budgets | Pinned offline adapter and `profile_phoneme_model.py` | Cache model weights and run on target CPU/GPU |
| Alignment and phoneme accuracy | Synthetic algorithm tests only | Human-aligned learner-speech corpus |
| Phoneme thresholds/FAR/FRR | Calibrator rejects insufficient/unfeasible data | Correct/error labels per phone |
| Random Forest training/comparison | Trainer smoke-tested on 500 synthetic rows; not a quality result | 500+ reviewed real attempts with speaker groups |
| A/B deployment and drift | Stable assignment and PSI implementation | Approved candidate plus production traffic |
| Word accuracy >95% | Validation match 18.75%, p95 63.55 ms, FAR 1.953% | Replace or retrain weak DTW identity baseline |
| Frontend functional acceptance | JS syntax, backend adapter and Python tests pass | Manual supported-browser microphone matrix |
| End-to-end phoneme latency | Identity measured; phoneme weights unavailable | Cached model on target hardware |
| Uptime, production p95 and data loss | Metrics, health, backup and resilience implemented | Staging/production soak test and SLO window |
| Compliance audit | Technical child-data review completed | Legal/DPO approval and operational evidence |
| Personnel/resources | Roles and minimum hardware documented | Project staffing/procurement decision |

Current SQLite evidence: 496 processed recordings across deterministic
speaker-disjoint train/validation/test splits, zero reviewed decision logs and
zero clinician reviews. Existing recordings are word labels, not annotated
phoneme errors.
