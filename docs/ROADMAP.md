# advX Implementation Roadmap

**Generated:** 2026-07-24
**Status:** Execution (see `IMPLEMENTATION_STATUS.md` for empirical blockers)

This roadmap consolidates:
- Architecture analysis from code review
- PRONUNCIATION_CORRECTION_DESIGN.md requirements
- Security, performance, and integration recommendations
- Frontend integration requirements

---

## Phase 0: Foundation & Technical Debt (1-2 weeks)

### 0.1 Security Hardening
- [x] Add rate limiting to API endpoints
- [x] Add input validation to WebSocket messages (max message size, schema validation)
- [x] Add authentication stub for WebSocket endpoints (even if placeholder)
- [x] Remove `.env` from repo, ensure only `.env.example` exists
- [x] Add `.env` to `.gitignore` verification
- [x] Add request logging with correlation IDs

### 0.2 Code Quality
- [x] Add type hints to all public functions (currently partial)
- [x] Add docstrings to `temporal_benchmark.py` complex functions
- [x] Remove `asyncio.to_thread` + `ThreadPoolExecutor` mixing in `streaming_inference.py` — choose one pattern
- [x] Add structured logging (JSON format) instead of print statements
- [x] Fix hardcoded `voice_rms_threshold = 0.008` in `StreamingConfig` — make it configurable/calibratable

### 0.3 Infrastructure
- [x] Add health check endpoint for SQLite (not just Redis)
- [x] Add graceful shutdown handling for WebSocket connections
- [x] Add Docker Compose profile for development vs production
- [x] Document minimum hardware requirements (RAM for DTW, model loading)

---

## Phase 1: Word Identity Gate (2-3 weeks)

**Goal:** Reliably distinguish target word from other vocabulary words

### 1.1 Model Selection Decision
- [x] **DECISION REQUIRED:** Choose word recognition approach:
  - Option A: Closed-vocabulary classifier (5 words = 5-class classifier)
  - Option B: Open-vocabulary ASR + word matching
  - Option C: Acoustic embedding similarity (current DTW extended)
- [x] Document decision with trade-offs in `docs/WORD_GATE_DECISION.md`
- [x] Define latency budget (< 100ms for word identity alone?)

### 1.2 Implementation
- [x] Create `app/word_identity.py` module
- [x] Define `WordIdentityResult` dataclass with:
  ```python
  @dataclass(frozen=True)
  class WordIdentityResult:
      target_word: str
      recognized_word: str | None
      target_probability: float
      alternatives: dict[str, float]  # top-k other words
      confidence: float
      model_version: str
  ```
- [x] Implement `WordIdentityGate` protocol
- [x] Add calibration for "mismatch confidence threshold"
- [x] Add tests for all word pairs in vocabulary (5 words = 20 ordered pairs)

### 1.3 Integration
- [x] Integrate into `streaming_inference.py` after quality gate
- [x] Add `word_identity` field to `stream.inference.partial` events
- [x] Add `WRONG_WORD` as new decision status variant

### 1.4 Success Criteria
- [x] False acceptance rate of wrong words < 2% on validation set
- [x] Word identity latency < 100ms
- [x] All 20 word-pair combinations tested

---

## Phase 2: Phoneme Model Integration (3-4 weeks)

**Goal:** Get frame-level phoneme probabilities from audio

### 2.1 Model Selection Decision
- [x] **DECISION REQUIRED:** Choose phoneme model:
  - Option A: wav2vec2-xlsr (multilingual, open license)
  - Option B: Whisper (larger, but robust)
  - Option C: Custom CTC model trained on TIMIT/LibriSpeech
  - Option D: External API (Google Speech-to-Text phonemes)
- [x] Document decision in `docs/PHONEME_MODEL_DECISION.md`
- [x] Define:
  - Inference location: local CPU/GPU vs. server
  - Memory budget
  - Latency budget
  - License compatibility

### 2.2 Implementation
- [x] Create `app/phoneme_model.py` module
- [x] Define protocol:
  ```python
  class PhonemeModel(Protocol):
      def infer(self, pcm_s16le: bytes, sample_rate: int) -> PhonemeEvidence:
          """Return frame-level phoneme probabilities."""
          ...
  
  @dataclass(frozen=True)
  class PhonemeEvidence:
      frames: np.ndarray  # shape: (n_frames, n_phonemes)
      phoneme_set: tuple[str, ...]
      frame_times_ms: list[float]
      model_version: str
  ```
- [x] Implement model loader with versioning
- [x] Add model download/caching mechanism
- [x] Create phoneme-to-IPA mapping for English

### 2.3 Lexicon Service
- [x] Create `app/lexicon.py` module
- [x] Integrate CMU Pronouncing Dictionary or similar
- [x] Support multiple acceptable pronunciations per word
- [x] Add stress markers preservation

### 2.4 Success Criteria
- [ ] Model loads in < 5 seconds on target hardware
- [ ] Inference latency < 200ms for 1-second audio
- [ ] Memory usage documented and within budget

---

## Phase 3: Forced Alignment (2-3 weeks)

**Goal:** Align phoneme probabilities with expected pronunciation

### 3.1 Implementation
- [x] Create `app/forced_alignment.py` module
- [x] Implement CTC-based alignment algorithm
- [x] Handle:
  - Insertions (extra sounds)
  - Deletions (missing phonemes)
  - Substitutions (wrong phonemes)
  - Duration anomalies
- [x] Support multiple pronunciation variants

### 3.2 Output Format
```python
@dataclass(frozen=True)
class AlignedPhoneme:
    expected: str
    observed: str
    start_ms: float
    end_ms: float
    expected_probability: float
    best_alternative: str
    best_alternative_probability: float
    alignment_confidence: float
    status: Literal["MATCH", "SUBSTITUTION", "INSERTION", "DELETION"]
```

### 3.3 Success Criteria
- [ ] Alignment accuracy > 90% on held-out correct pronunciations
- [ ] Correctly identifies > 80% of annotated phoneme errors

---

## Phase 4: Phoneme Scoring & Calibration (2-3 weeks)

**Goal:** Calibrated per-phoneme scores with meaningful thresholds

### 4.1 Data Collection
- [ ] Collect or acquire learner speech with phoneme error annotations
- [x] Split by speaker: train/validation/test
- [x] Freeze test set — never use for threshold tuning

### 4.2 Scoring Implementation
- [x] Create `app/phoneme_scorer.py` module
- [x] Implement score formula:
  ```python
  phone_score = log P(expected | audio) - log P(best_alternative | audio)
  ```
- [x] Add duration normalization
- [x] Add per-phoneme calibration thresholds

### 4.3 Calibration
- [ ] Compute thresholds on validation set
- [ ] Target metrics:
  - False acceptance rate of phoneme errors < 10%
  - False rejection rate of correct phonemes < 5%
- [x] Document calibration procedure for reproducibility

### 4.4 Success Criteria
- [ ] Per-phoneme accuracy > 85% on test set
- [ ] No phoneme has > 20% error rate

---

## Phase 5: Decision Engine Refactor (1-2 weeks)

**Goal:** Unified decision engine with rule-based baseline and ML-ready interface

### 5.1 Implementation
- [x] Create `app/decision_engine.py` module
- [x] Implement rule-based engine following PRONUNCIATION_CORRECTION_DESIGN.md section 4.8
- [x] Add interface for future ML-based decisions:
  ```python
  class DecisionEngine(Protocol):
      def decide(self, signals: DecisionSignals) -> DecisionResult:
          ...
  
  @dataclass
  class DecisionSignals:
      quality: QualityResult
      word_identity: WordIdentityResult | None
      phoneme_scores: list[PhonemeScore] | None
      dtw_distance: float | None
      # ... other signals
  ```
- [x] Add logging of all signals for every decision

### 5.2 Threshold Learning (Interim before Random Forest)
- [x] Implement grid search for optimal thresholds
- [x] Use Bayesian optimization for multi-threshold tuning
- [x] Document threshold values in versioned config

### 5.3 Random Forest Preparation
- [x] Design feature vector schema for ML model
- [x] Create signal logging format suitable for ML training:
  ```json
  {
    "attempt_id": "...",
    "signals": {
      "word_identity_confidence": 0.91,
      "dtw_distance": 0.23,
      "phoneme_score_mean": 0.78,
      "phoneme_score_min": 0.45,
      "alignment_quality": 0.85,
      "duration_ms": 450,
      "snr_db": 21.4
    },
    "ground_truth": null  // filled by human reviewer
  }
  ```
- [x] Create annotation interface for human reviewers

---

## Phase 6: Frontend Integration (2-3 weeks)

**Goal:** Connect existing frontend to backend APIs

### 6.1 API Contract Finalization
- [x] Document all WebSocket message formats in OpenAPI/AsyncAPI
- [x] Create frontend integration guide in `docs/FRONTEND_INTEGRATION.md`
- [x] Define session lifecycle:
  - Session creation
  - Audio streaming
  - Real-time feedback
  - Final result
  - Session termination

### 6.2 WebSocket Endpoints
- [x] Verify `/sessions/{session_id}/events` matches frontend expectations
- [x] Verify `/streaming/sessions/{session_id}` matches frontend expectations
- [x] Add connection error handling and reconnection logic
- [x] Add heartbeat/ping-pong for connection health

### 6.3 Event Format Alignment
- [x] Align `stream.inference.partial` format with frontend needs
- [x] Align `stream.inference.final` format with frontend needs
- [x] Add `pronunciation.evaluated` event for final decisions
- [x] Add internationalization support for messages (Italian/English)

### 6.4 Frontend Developer Handoff
- [x] Provide Postman/Insomnia collection for testing
- [x] Provide example WebSocket client code
- [x] Document deployment requirements (CORS, WebSocket proxy if needed)

---

## Phase 7: Production Readiness (2-3 weeks)

### 7.1 Observability
- [x] Add Prometheus metrics endpoint
- [x] Add structured logging with trace IDs
- [x] Add decision audit trail in SQLite
- [x] Add performance monitoring (latency percentiles)

### 7.2 Error Handling
- [x] Add circuit breaker for Redis connections
- [x] Add fallback behavior when phoneme model unavailable
- [x] Add graceful degradation when model inference fails
- [x] Add retry logic with exponential backoff

### 7.3 Deployment
- [x] Create production Dockerfile
- [x] Add Kubernetes manifests (optional)
- [x] Add environment variable validation on startup
- [x] Add startup health checks
- [x] Document backup/restore for SQLite

### 7.4 Security
- [x] Add TLS termination documentation
- [x] Add authentication/authorization layer
- [x] Add rate limiting per session
- [x] Add input sanitization for all user inputs
- [x] Security audit for child data handling

---

## Phase 8: ML Enhancement (Future)

**Prerequisites:** 500+ labeled attempts with ground truth

### 8.1 Random Forest Decision Engine
- [ ] Train Random Forest on labeled data
- [ ] Compare against rule-based baseline
- [ ] Deploy with A/B testing
- [ ] Monitor for drift

### 8.2 Continuous Learning
- [x] Design feedback loop from human reviews
- [x] Implement model versioning and rollback
- [ ] Schedule periodic retraining

---

## Decision Points Summary

| Decision | Status | Owner | Deadline |
|----------|--------|-------|----------|
| Word identity model | **REQUIRED** | - | Before Phase 1 |
| Phoneme model | **REQUIRED** | - | Before Phase 2 |
| Local vs. server inference | **REQUIRED** | - | Before Phase 2 |
| Latency/memory budget | **REQUIRED** | - | Before Phase 2 |
| Initial vocabulary | Defined (5 words) | - | Done |
| Target population | Pediatric? | - | Clarify |
| False acceptance tolerance | **REQUIRED** | - | Before Phase 4 |

---

## Success Metrics

### MVP Success (end of Phase 6)
- [ ] Distinguishes target word from other vocabulary words with > 95% accuracy
- [ ] Detects annotated phoneme errors with > 80% accuracy
- [x] Never returns `CORRECT` without positive evidence
- [x] All decisions explainable with confidence and reason codes
- [ ] Frontend successfully integrated and functional
- [ ] End-to-end latency < 500ms for 1-second audio

### Production Success (end of Phase 7)
- [ ] Uptime > 99.5%
- [ ] P95 latency < 1 second
- [ ] Zero data loss on failures
- [ ] Complete audit trail for compliance

---

## Resource Requirements

### Data
- [ ] 200-500 labeled correct/incorrect pronunciations for threshold calibration
- [ ] 500+ labeled attempts for Random Forest training (Phase 8)
- [x] Speaker-disjoint train/validation/test splits

### Hardware
- [x] Minimum: 4GB RAM for SQLite + DTW
- [ ] With phoneme model: 8GB RAM minimum, 16GB recommended
- [ ] GPU optional but recommended for model inference

### Personnel
- [ ] Backend engineer: phases 0-3, 5-7
- [ ] ML/Audio engineer: phases 2-4, 8
- [ ] Frontend engineer: phase 6
- [ ] DevOps: phase 7
- [ ] Domain expert (logopedist): data annotation, threshold validation

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 0. Foundation | 1-2 weeks | 2 weeks |
| 1. Word Identity | 2-3 weeks | 5 weeks |
| 2. Phoneme Model | 3-4 weeks | 9 weeks |
| 3. Forced Alignment | 2-3 weeks | 12 weeks |
| 4. Phoneme Scoring | 2-3 weeks | 15 weeks |
| 5. Decision Engine | 1-2 weeks | 17 weeks |
| 6. Frontend Integration | 2-3 weeks | 20 weeks |
| 7. Production Ready | 2-3 weeks | 23 weeks |

**MVP Target:** 20 weeks (5 months)
**Production Ready:** 23 weeks (6 months)

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Phoneme model too slow | Medium | High | Profile early; consider quantization |
| Insufficient labeled data | High | High | Start collection now; consider synthetic |
| Frontend integration friction | Medium | Medium | Early API contract testing |
| DTW thresholds don't generalize | Medium | Medium | Treat as signal, not decision |
| Model license incompatibility | Low | High | Check licenses before integration |

---

## Next Actions (This Week)

1. [x] Create `docs/WORD_GATE_DECISION.md` and document word identity approach
2. [x] Create `docs/PHONEME_MODEL_DECISION.md` and research options
3. [x] Add rate limiting to API endpoints
4. [x] Fix `voice_rms_threshold` to be configurable
5. [x] Add structured logging
6. [x] Create decision log format for future ML training
