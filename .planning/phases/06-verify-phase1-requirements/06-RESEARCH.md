# Phase 6: Verify Phase 1 Requirements - Research

**Researched:** 2026-03-10
**Domain:** Verification scripting, Python data pipeline validation, gap closure
**Confidence:** HIGH

## Summary

Phase 6 is a verification and gap closure phase, not a feature-building phase. The work involves: (1) creating a standalone verification script at `scripts/verify_phase1.py` that exercises each Phase 1 requirement (DATA-01, DATA-02, DATA-03, DATA-05, INFR-01) with both synthetic and real CICIDS2017 data, (2) fixing the missing `select_features` re-export from `data/__init__.py`, (3) adding a regression test for the re-export, and (4) producing a thesis-appendix-ready VERIFICATION.md in the phase directory.

The codebase is mature and well-tested -- 89 existing tests pass across 4 test files (test_config.py, test_loader.py, test_preprocess.py, test_partition.py). All Phase 1 code (loader, preprocess, partition, config) is structurally complete and functionally correct. The gap is purely procedural: no VERIFICATION.md was created during Phase 1, leaving 5 requirements "orphaned" per the v1.0 Milestone Audit. The verification script must run the actual code paths and capture measured values as evidence.

**Primary recommendation:** Build a single `scripts/verify_phase1.py` script with one check function per requirement, producing human-readable stdout output. Fix the `select_features` re-export first (it is a prerequisite for DATA-02 verification). Generate VERIFICATION.md from the script output plus manual formatting.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Verification evidence:**
- Pass/fail with actual measured values per requirement (e.g. "Feature count: 32, range 20-40: PASS")
- Traceability table in VERIFICATION.md mapping each requirement ID to its check, result, and value
- Brief method line per check documenting how it was verified (e.g. "Verified via: scripts/verify_phase1.py")
- Record verification date and environment (Python version, OS) in VERIFICATION.md header

**Test data source:**
- Both synthetic and real CICIDS2017 data
- Synthetic tests verify code paths and logic
- Real CICIDS2017 run captures actual numeric results (row counts, feature counts, class distributions) for thesis-grade evidence
- Graceful fallback: if CICIDS2017 download fails/unavailable, real data section marked "SKIPPED -- data not available" and phase still passes on synthetic results alone
- CICIDS2017 needs to be downloaded first (not yet available locally)

**Verification approach:**
- Standalone verification script at `scripts/verify_phase1.py`
- Script does its own requirement-specific checks (load data, count features, check scaler, etc.)
- Human-readable output to stdout, captured into VERIFICATION.md
- Existing pytest tests run separately -- script does not invoke pytest
- Script must handle missing real data gracefully (fallback to synthetic-only)

**select_features fix:**
- Add `select_features` re-export to `data/__init__.py` and update `__all__`
- Add a test that imports `select_features` from `federated_ids.data` to prevent future regressions
- Verification script calls `select_features` standalone on a small DataFrame to prove the re-export is functional
- Claude's discretion: check preprocess.py for other useful public functions that should also be re-exported

### Claude's Discretion

- Which other public functions from preprocess.py (if any) to add to data/__init__.py exports
- Internal structure of the verification script (how checks are organized)
- Exact format of the human-readable output
- How to generate/download synthetic test data within the script

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Load CICIDS2017 CSV files and clean data (handle inf/NaN values, whitespace column names, constant columns) | `loader.py` has validation gates (assertions) for Inf/NaN/labels. Verification script calls `load_cicids2017()` on synthetic CSV, checks zero Inf/NaN in result. Code at lines 122-134 of loader.py. |
| DATA-02 | Select and engineer features (reduce 78+ raw features to 20-40 informative ones) | `select_features()` in preprocess.py lines 114-260. Verify by calling on synthetic data and checking `len(feature_names)` is in 20-40 range. Also must fix re-export from `data/__init__.py`. |
| DATA-03 | Normalize features with StandardScaler fitted on training data only (no data leakage) | `preprocess()` at lines 321-324 fits scaler on X_train only, transforms both. Verification: check `X_train.mean(axis=0)` is near 0, `X_test.mean(axis=0)` is NOT near 0. |
| DATA-05 | Partition data IID across 2-5 clients with stratified splits maintaining class ratios | `partition_iid()` in partition.py uses StratifiedKFold. Verify: partition with num_clients in {2,3,5}, check each partition's class ratio is within 5% of global ratio. |
| INFR-01 | Configuration file (YAML/JSON) for all hyperparameters (LR, epochs, batch size, FL rounds, num clients) | `config.py` with `load_config()` + `config/default.yaml`. Verify: load config, check all required keys exist (learning_rate, local_epochs, batch_size, num_rounds, num_clients). |

</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | >=3.11 | Runtime | Project requirement in pyproject.toml |
| pandas | >=2.2.0 | DataFrame operations for synthetic data generation | Already installed, used throughout data pipeline |
| numpy | >=1.26.0 | Numeric operations, array validation | Already installed |
| scikit-learn | >=1.5.0 | StandardScaler, StratifiedKFold verification | Already installed |
| PyYAML | >=6.0 | Config loading verification | Already installed |
| pytest | >=8.0.0 | Regression test for select_features re-export | Already installed as dev dependency |

### Supporting

No new libraries needed. This phase uses only what is already installed.

### Alternatives Considered

None -- this is a verification phase, not a feature phase. No new dependencies required.

**Installation:**
```bash
# No new packages needed -- all dependencies already in pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure

```
scripts/
  verify_phase1.py          # NEW: standalone verification script
tests/
  test_preprocess.py         # MODIFY: add select_features import test
src/federated_ids/data/
  __init__.py                # MODIFY: add select_features to exports
.planning/phases/06-verify-phase1-requirements/
  06-VERIFICATION.md         # NEW: formal verification report
```

### Pattern 1: Verification Script Structure

**What:** A single Python script with one check function per requirement, a summary table printer, and a main entry point that runs all checks.
**When to use:** Whenever requirements need formal verification with measured evidence.
**Example:**
```python
import sys
import platform
from datetime import datetime

def check_data_01():
    """DATA-01: CSV loading and cleaning produces zero Inf/NaN."""
    # Generate synthetic CSV or load real data
    # Call load_cicids2017()
    # Assert no Inf/NaN
    # Return dict with {status, value, method}
    pass

def check_data_02():
    """DATA-02: Feature selection reduces columns to 20-40 range."""
    pass

# ... one function per requirement

def main():
    print(f"Verification Date: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"OS: {platform.platform()}")

    results = {}
    for check_fn in [check_data_01, check_data_02, ...]:
        result = check_fn()
        results[result["req_id"]] = result

    # Print summary table
    print_summary_table(results)

    # Return exit code based on pass/fail
    all_pass = all(r["status"] == "PASS" for r in results.values())
    sys.exit(0 if all_pass else 1)
```

### Pattern 2: Synthetic Data Generation Within Script

**What:** The script generates its own synthetic data inline rather than depending on external fixtures or conftest.py.
**When to use:** When the script must be independently runnable (`python scripts/verify_phase1.py`) without pytest.
**Example:**
```python
def _make_synthetic_csv(tmp_dir):
    """Create a minimal synthetic CICIDS2017 CSV for verification."""
    rng = np.random.RandomState(42)
    n_rows = 200

    # Must include columns that appear in _DOMAIN_SHORTLIST
    # so select_features has enough to work with
    data = {}
    for col in EXPECTED_COLUMNS:
        data[col] = rng.rand(n_rows) * 1000

    # Inject Inf/NaN to verify cleaning
    data["Flow Bytes/s"][0] = np.inf
    data["Flow Bytes/s"][1] = np.nan

    # Labels: mix of BENIGN and DDoS
    labels = ["BENIGN"] * 120 + ["DDoS"] * 80
    data[" Label"] = labels  # leading whitespace on column name

    df = pd.DataFrame(data)
    csv_path = os.path.join(tmp_dir, "synthetic_test.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
```

### Pattern 3: Graceful Real Data Fallback

**What:** Check if real CICIDS2017 CSV exists at the configured path. If not, mark that section as SKIPPED.
**When to use:** When verification should work without external data but benefits from real data when available.
**Example:**
```python
def _check_real_data_available(config):
    """Check if real CICIDS2017 data exists."""
    raw_dir = config["data"]["raw_dir"]
    files = config["data"]["files"]
    for f in files:
        if not os.path.isfile(os.path.join(raw_dir, f)):
            return False
    return True

# In each check function:
if real_data_available:
    # Run with real data, capture actual values
    result["real_data"] = {"row_count": len(df), ...}
else:
    result["real_data"] = "SKIPPED -- data not available"
```

### Anti-Patterns to Avoid

- **Invoking pytest from the verification script:** The script does its own checks independently. Existing pytest tests run separately.
- **Modifying source code beyond the select_features fix:** This is a verification phase. Do not refactor or add features.
- **Hard-coding expected values from synthetic data:** Use range checks (e.g., "20-40 features") rather than exact counts, since synthetic data may produce slightly different feature counts depending on correlation patterns.
- **Catching and suppressing exceptions:** If a code path fails, that is a verification failure. Let it propagate and report as FAIL.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Synthetic CICIDS2017 data | New fixture framework | Reuse column list from `conftest.py` (`_CICIDS_COLUMNS`) as reference, but generate independently in script | Consistency with existing test fixtures |
| YAML config validation | Custom config checker | Call `load_config("config/default.yaml")` directly | The function already validates all required keys |
| Scaler leakage detection | Custom statistics | `np.allclose(X_train.mean(axis=0), 0, atol=1e-6)` for train, `not np.allclose(X_test.mean(axis=0), 0, atol=0.05)` for test | Same approach used in existing `test_preprocess.py` |

**Key insight:** The verification script should call the actual production code paths and inspect results, not re-implement the logic.

## Common Pitfalls

### Pitfall 1: Synthetic Data Too Small for Feature Selection

**What goes wrong:** With too few rows or columns, `select_features` may drop too many features to fall in the 20-40 range, or the domain shortlist fallback triggers unexpectedly.
**Why it happens:** The domain shortlist has 44 features. If the synthetic data does not include enough of these column names, the function falls back to all numeric columns.
**How to avoid:** Generate synthetic data with column names matching `_DOMAIN_SHORTLIST` from preprocess.py (all 44 features). Use at least 200 rows to ensure statistical filters (variance, correlation) behave meaningfully.
**Warning signs:** Feature count outside 20-40 range on synthetic data.

### Pitfall 2: Scaler Leakage Test False Positive

**What goes wrong:** The "test mean is NOT near 0" check can occasionally pass even with leakage if the test set happens to have means near 0 by chance.
**Why it happens:** Small synthetic datasets have higher variance in feature means.
**How to avoid:** Use enough samples (200+ per split) and check that at least some feature means deviate from 0, rather than requiring ALL to deviate.
**Warning signs:** Test passes/fails non-deterministically.

### Pitfall 3: Real Data Path Assumptions

**What goes wrong:** The verification script assumes `data/raw/` contains the CICIDS2017 CSV, but the file might be elsewhere or named differently.
**Why it happens:** Different users may download different mirrors or rename files.
**How to avoid:** Read the path from `config/default.yaml` using `load_config()`. Check the exact filename configured in `data.files`.
**Warning signs:** FileNotFoundError on real data path.

### Pitfall 4: Forgetting to Update __all__ in __init__.py

**What goes wrong:** Adding the import of `select_features` but not updating `__all__` means `from federated_ids.data import *` still does not include it.
**Why it happens:** Easy to add the import line and forget `__all__`.
**How to avoid:** Update both the import statement AND the `__all__` list. The regression test should import from the package, not the module.

### Pitfall 5: VERIFICATION.md Format Inconsistency

**What goes wrong:** The verification report does not match the format used by Phase 2/3/5 VERIFICATION.md files.
**Why it happens:** Different phases used slightly different formats.
**How to avoid:** Follow the format established in `02-VERIFICATION.md` (the most detailed one) with frontmatter, traceability table, observable truths, and requirements coverage sections.

## Code Examples

### Verified: select_features Re-export Fix

```python
# In src/federated_ids/data/__init__.py
# Add import:
from federated_ids.data.preprocess import select_features

# Update __all__:
__all__ = [
    "load_cicids2017",
    "preprocess",
    "run_pipeline",
    "partition_iid",
    "create_dataloaders",
    "select_features",  # ADD THIS
]
```

### Verified: Regression Test for Re-export

```python
# In tests/test_preprocess.py (add new test class)
class TestSelectFeaturesReExport:
    """Regression test: select_features is importable from data package."""

    def test_import_from_package(self):
        """select_features can be imported from federated_ids.data."""
        from federated_ids.data import select_features
        assert callable(select_features)
```

### Verified: DATA-01 Check Pattern

```python
def check_data_01(df):
    """DATA-01: CSV loading produces zero Inf/NaN values."""
    numeric_df = df.select_dtypes(include=[np.number])
    n_inf = np.isinf(numeric_df.values).sum()
    n_nan = np.isnan(numeric_df.values).sum()
    status = "PASS" if (n_inf == 0 and n_nan == 0) else "FAIL"
    return {
        "req_id": "DATA-01",
        "check": "Zero Inf/NaN after cleaning",
        "status": status,
        "value": f"Inf: {n_inf}, NaN: {n_nan}",
        "method": "load_cicids2017() + np.isinf/isnan check",
    }
```

### Verified: DATA-03 Scaler Leakage Check Pattern

```python
def check_data_03(result):
    """DATA-03: StandardScaler fitted on training data only."""
    train_means = np.mean(result["X_train"], axis=0)
    test_means = np.mean(result["X_test"], axis=0)

    train_near_zero = np.allclose(train_means, 0, atol=1e-6)
    test_not_near_zero = not np.allclose(test_means, 0, atol=0.05)

    status = "PASS" if (train_near_zero and test_not_near_zero) else "FAIL"
    return {
        "req_id": "DATA-03",
        "check": "StandardScaler fitted on training data only (no leakage)",
        "status": status,
        "value": f"Train mean~0: {train_near_zero}, Test mean!=0: {test_not_near_zero}",
        "method": "preprocess() + np.allclose on feature means",
    }
```

### Verified: INFR-01 Config Check Pattern

```python
def check_infr_01():
    """INFR-01: YAML config controls all hyperparameters."""
    from federated_ids.config import load_config
    config = load_config("config/default.yaml")

    required_params = {
        "training.learning_rate": config["training"]["learning_rate"],
        "training.local_epochs": config["training"]["local_epochs"],
        "training.batch_size": config["training"]["batch_size"],
        "federation.num_rounds": config["federation"]["num_rounds"],
        "federation.num_clients": config["federation"]["num_clients"],
    }

    missing = [k for k, v in required_params.items() if v is None]
    status = "PASS" if len(missing) == 0 else "FAIL"
    return {
        "req_id": "INFR-01",
        "check": "YAML config contains all hyperparameters",
        "status": status,
        "value": f"Found {len(required_params)} required params, missing: {missing}",
        "method": "load_config('config/default.yaml') + key presence check",
    }
```

### Verified: VERIFICATION.md Traceability Table Format

From CONTEXT.md user decision:
```markdown
| REQ ID | Check | Method | Result | Value |
|--------|-------|--------|--------|-------|
| DATA-01 | Zero Inf/NaN after cleaning | scripts/verify_phase1.py | PASS | Inf: 0, NaN: 0 |
| DATA-02 | Feature count in 20-40 range | scripts/verify_phase1.py | PASS | Feature count: 32 |
| DATA-03 | Scaler fitted on train only | scripts/verify_phase1.py | PASS | Train mean~0: True, Test mean!=0: True |
| DATA-05 | IID partitioning with stratified splits | scripts/verify_phase1.py | PASS | 3 clients, class ratio deviation <5% |
| INFR-01 | YAML config has all hyperparams | scripts/verify_phase1.py | PASS | 5/5 required params present |
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual code inspection | Automated verification scripts | Project decision (Phase 6) | Produces re-runnable, thesis-appendix-ready evidence |

**Deprecated/outdated:**
- None relevant -- this phase uses only established Python standard library and existing project dependencies.

## Additional Research: preprocess.py Public Functions

**Discretion item:** Check preprocess.py for other public functions that should be re-exported.

Public functions in `preprocess.py`:
1. `select_features(df, config)` -- PUBLIC, useful independently, **should be re-exported** (this is the fix)
2. `preprocess(df, config)` -- already re-exported
3. `main(config_path)` -- already re-exported as `run_pipeline`

Private/internal:
- `_cache_exists(processed_dir)` -- prefixed with underscore, internal helper
- `_DOMAIN_SHORTLIST` -- module constant, not a function
- `_CACHE_FILES` -- module constant, not a function

**Recommendation:** Only `select_features` needs to be added. No other public functions are missing from the re-exports. The constants prefixed with `_` are intentionally private.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >=8.0.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | CSV loading produces zero Inf/NaN | verification script | `python scripts/verify_phase1.py` | No -- Wave 0 (script) |
| DATA-02 | Feature selection reduces to 20-40 columns; select_features re-exported | verification script + unit test | `python -m pytest tests/test_preprocess.py::TestSelectFeaturesReExport -x` | No -- Wave 0 (test class) |
| DATA-03 | StandardScaler fitted on train only, no leakage | verification script | `python scripts/verify_phase1.py` | No -- Wave 0 (script) |
| DATA-05 | IID partitioning across configurable clients with stratified splits | verification script | `python scripts/verify_phase1.py` | No -- Wave 0 (script) |
| INFR-01 | YAML config controls all hyperparameters | verification script | `python scripts/verify_phase1.py` | No -- Wave 0 (script) |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/ -x -q` (existing tests still pass)
- **Per wave merge:** `python -m pytest tests/ -v && python scripts/verify_phase1.py` (full suite + verification)
- **Phase gate:** All pytest tests green + verification script exits 0 before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `scripts/verify_phase1.py` -- standalone verification script covering DATA-01, DATA-02, DATA-03, DATA-05, INFR-01
- [ ] `tests/test_preprocess.py::TestSelectFeaturesReExport` -- regression test for select_features import from package
- [ ] `src/federated_ids/data/__init__.py` -- fix: add select_features re-export (code change, not test gap)

## Open Questions

1. **CICIDS2017 Download Availability**
   - What we know: The real data file is `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` from CICIDS2017. It is not currently downloaded locally.
   - What's unclear: Whether the download will succeed (multiple mirrors exist, some are unreliable per STATE.md blocker note).
   - Recommendation: The verification script handles this gracefully via the locked decision -- synthetic results are sufficient for PASS, real data results are bonus evidence. The script should attempt to use real data from the configured path but mark as SKIPPED if unavailable.

2. **Exact Feature Count on Synthetic Data**
   - What we know: The domain shortlist has 44 features. After statistical filtering (zero-variance, near-constant, correlation), the count drops.
   - What's unclear: Exactly how many features survive filtering on synthetic data depends on the random seed and data generation approach.
   - Recommendation: Generate synthetic data with enough variance and diversity in the 44 domain shortlist columns. The check should validate the count is in the 20-40 range, not an exact number.

## Sources

### Primary (HIGH confidence)
- Source code inspection of: `data/__init__.py`, `data/loader.py`, `data/preprocess.py`, `data/partition.py`, `config.py`
- Source code inspection of: `config/default.yaml`
- Source code inspection of: `tests/conftest.py`, `tests/test_loader.py`, `tests/test_preprocess.py`, `tests/test_partition.py`, `tests/test_config.py`
- `.planning/v1.0-MILESTONE-AUDIT.md` -- gap analysis identifying 5 orphaned Phase 1 requirements
- `.planning/phases/02-model-definition-and-local-training/02-VERIFICATION.md` -- reference format for verification reports
- Runtime verification: `python -m pytest tests/ -x -q` -- 89 tests pass
- Runtime verification: `from federated_ids.data import select_features` -- confirmed ImportError

### Secondary (MEDIUM confidence)
- None needed -- all findings verified from source code

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, everything already installed and tested
- Architecture: HIGH -- verification script pattern is straightforward Python scripting; reference format exists in Phase 2 VERIFICATION.md
- Pitfalls: HIGH -- identified from direct code inspection and test analysis; synthetic data concerns verified by examining conftest.py fixtures

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable -- no external dependencies changing)
