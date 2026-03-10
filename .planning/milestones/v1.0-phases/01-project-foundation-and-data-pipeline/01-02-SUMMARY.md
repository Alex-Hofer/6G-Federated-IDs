---
phase: 01-project-foundation-and-data-pipeline
plan: 02
subsystem: data
tags: [pandas, sklearn, StandardScaler, feature-selection, CICIDS2017, preprocessing, class-weights]

# Dependency graph
requires:
  - phase: 01-01
    provides: project scaffold, config system, seed utility, data subpackage init
provides:
  - CSV data loader with CICIDS2017-specific cleaning (whitespace, Inf/NaN, label mapping)
  - Hybrid feature selection (domain shortlist + statistical filters)
  - StandardScaler normalization fitted on training data only (no leakage)
  - Stratified global train/test split (80/20)
  - Class weight computation and artifact persistence
  - 23 tests covering loader and preprocessing pipeline
affects: [01-03, phase-2 model training, phase-3 federated partitioning]

# Tech tracking
tech-stack:
  added: [joblib, sklearn.preprocessing.StandardScaler, sklearn.model_selection.train_test_split, sklearn.utils.class_weight.compute_class_weight]
  patterns: [three-stage pipeline with validation gates, domain shortlist + statistical filter hybrid feature selection, scaler fit-on-train-only pattern]

key-files:
  created:
    - src/federated_ids/data/loader.py
    - src/federated_ids/data/preprocess.py
    - tests/test_loader.py
    - tests/test_preprocess.py
  modified:
    - tests/conftest.py

key-decisions:
  - "Domain shortlist of 44 DDoS-relevant features with fallback to all numeric columns if fewer than target remain"
  - "Near-constant filter at >99% same value threshold in addition to zero-variance filter"
  - "Correlation filtering keeps the feature with higher variance from each correlated pair"
  - "Class weights saved as JSON (portable) alongside scaler saved as joblib pkl"

patterns-established:
  - "Validation gates: assert-based checks at pipeline boundaries (no Inf, no NaN, correct dtypes)"
  - "Feature drop report: dict mapping dropped feature names to {reason: str} for auditability"
  - "Scaler fit-on-train-only: fit_transform on X_train, transform-only on X_test"
  - "Artifact persistence: scaler.pkl, features.json, class_weights.json, class_distribution.json to processed_dir"

requirements-completed: [DATA-01, DATA-02, DATA-03, DATA-04]

# Metrics
duration: 5min
completed: 2026-03-09
---

# Phase 1 Plan 02: Data Loading and Preprocessing Pipeline Summary

**CICIDS2017 loader with Inf/NaN cleaning and binary label mapping, plus hybrid feature selection (44-feature domain shortlist + statistical filters), StandardScaler with no data leakage, and class weight computation**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-09T14:06:14Z
- **Completed:** 2026-03-09T14:07:42Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Built CSV loader handling CICIDS2017 quirks: column whitespace stripping, Inf/NaN row removal, and multi-class to binary label mapping (BENIGN=0, DDoS variants=1)
- Implemented hybrid feature selection with 44-feature domain shortlist of DDoS-relevant network flow features, plus statistical filters (zero-variance, near-constant >99%, correlation >0.95)
- StandardScaler fitted on training data only with stratified 80/20 train/test split, preventing data leakage
- Class weight computation via sklearn and full artifact persistence (scaler, feature list, class weights, class distribution stats)
- Comprehensive test suite: 11 loader tests + 12 preprocessing tests, all using synthetic data fixtures (no real CICIDS2017 CSV required)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement data loader with CSV loading, cleaning, and label mapping** - `cccf86a` (feat)
2. **Task 2: Implement feature selection, normalization, train/test split, and class weight computation** - `8e37bf8` (feat)

## Files Created/Modified
- `src/federated_ids/data/loader.py` - CSV loading, column stripping, Inf/NaN removal, binary label mapping with validation gates
- `src/federated_ids/data/preprocess.py` - Hybrid feature selection, StandardScaler normalization, stratified split, class weights, artifact saving
- `tests/test_loader.py` - 11 tests: column cleaning, Inf/NaN removal, label mapping, error handling, return type
- `tests/test_preprocess.py` - 12 tests: identifier removal, zero-variance, correlation, feature count, split ratio, stratification, scaler leakage, dtype, class weights, artifacts, reproducibility
- `tests/conftest.py` - Updated in Plan 01-01 with CICIDS2017 synthetic data fixtures (sample_cicids_df, sample_csv_file) used by both test files

## Decisions Made
- Domain shortlist includes 44 features across 5 categories (flow-level, inter-arrival time, packet size, flag, rate/header) -- falls back to all numeric columns if fewer than target_features survive domain filtering
- Near-constant filter (>99% same value) added as separate step beyond zero-variance, catching features that technically have non-zero variance but are practically useless
- When dropping one of a correlated pair, the feature with higher variance is kept (more informative)
- Artifacts saved as JSON for portability (features, class weights, distribution stats) except scaler which uses joblib pkl (sklearn standard)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Python is not installed on this system. Tests are written but cannot be executed yet. User must install Python 3.11+ and run `pip install -e ".[dev]" && python -m pytest tests/ -x -v` to verify all 23 tests pass.

## User Setup Required

None - no external service configuration required. However, Python 3.11+ must be installed before running the project:
```bash
# After installing Python 3.11+
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -e ".[dev]"
python -m pytest tests/test_loader.py tests/test_preprocess.py -x -v
```

## Next Phase Readiness
- Data loader and preprocessing pipeline are complete, ready for Plan 01-03 (IID partitioning and DataLoader creation)
- loader.py exports: load_cicids2017, DDOS_LABELS, IDENTIFIER_COLS
- preprocess.py exports: preprocess, select_features
- Artifacts (scaler, feature list, class weights) will be available in data/processed/ for downstream consumption
- Global train/test split established before any client partitioning (Plan 01-03)

## Self-Check: PASSED

- All 5 files exist on disk (loader.py, preprocess.py, test_loader.py, test_preprocess.py, conftest.py)
- Both commits verified: cccf86a (Task 1), 8e37bf8 (Task 2)
- Line count minimums met: loader.py=142 (min 50), preprocess.py=405 (min 100), test_loader.py=98 (min 40), test_preprocess.py=275 (min 60)

---
*Phase: 01-project-foundation-and-data-pipeline*
*Completed: 2026-03-09*
