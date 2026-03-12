# Phase 8: Security & Integrity - Research

**Researched:** 2026-03-12
**Domain:** Input validation, serialization safety, dependency hygiene, config hardening
**Confidence:** HIGH

## Summary

Phase 8 addresses nine security and integrity requirements (SEC-01 through SEC-09) spanning four distinct domains: (1) replacing `assert`-based validation with production-safe `if/raise` patterns, (2) adding input validation to `fedavg_aggregate`, (3) replacing pickle serialization with safe JSON for the StandardScaler, and (4) hardening configuration handling (log_level validation, config_path validation, eliminating config dict mutation, dependency cleanup).

The codebase is a Python 3.11+ federated learning IDS project using PyTorch, scikit-learn, pandas, and numpy. It has 22 source files across 6 subpackages and 11 test files with comprehensive fixtures in `conftest.py`. The changes are purely internal refactoring -- no new dependencies are needed, and one dependency (`joblib`) will be removed entirely.

**Primary recommendation:** Organize implementation into three waves: (1) exception infrastructure + assert replacements, (2) fedavg validation + scaler serialization migration, (3) config hardening + dependency cleanup + tests. Each wave is independently testable.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Error handling style:**
- Use a custom `DataValidationError` exception (subclass of `ValueError`) for all data validation failures
- Define `DataValidationError` in a new `src/federated_ids/exceptions.py` module at the package root
- Error messages must be detailed with actual vs expected values (e.g., "Expected labels {0, 1}, got {0, 1, 2}" or "Found 42 NaN values in X_train")
- Do NOT log before raising -- just raise with a descriptive message; callers handle logging if needed
- Replace all 10 assert statements: loader.py (3), preprocess.py (6), partition.py (1)

**Scaler serialization:**
- Replace `joblib.dump(scaler, scaler_path)` with JSON serialization
- Use versioned JSON envelope format: `{"version": "1.0", "type": "StandardScaler", "params": {"mean_": [...], "scale_": [...], "var_": [...], "n_features_in_": 78}}`
- File name: `scaler.json` (replaces `scaler.pkl`)
- On load: strict validation -- verify array lengths match `n_features_in_`, all values are finite
- Migration: if only `.pkl` exists, raise error with message: "Run `federated-ids-preprocess` to regenerate scaler in safe JSON format" -- no auto-conversion, no pickle fallback
- After migration, `joblib` import can be removed from preprocess.py (SEC-05 becomes: don't need to declare joblib at all)

**FedAvg validation strictness:**
- Empty results list: raise immediately with clear error ("fedavg_aggregate received empty results -- no clients participated")
- Zero-example clients: skip with warning log, exclude from aggregation. Error only if ALL clients have zero examples after filtering
- NaN detection: check input parameters BEFORE aggregation (validate each client's submitted weights)
- NaN action: skip client with warning log ("Client X submitted NaN parameters, skipping"), continue with remaining valid clients. Error only if ALL clients are invalid after filtering

**Config safety:**
- Validate `log_level` against allowlist: {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}. Raise `ValueError` if invalid, showing the bad value and allowed options
- Validate `config_path` extension: must end with `.yaml` or `.yml`. Raise `ValueError` if not (existence already checked by `load_config`)
- Eliminate `config["_device"]` mutation in `server.py:462`: pass `device` as an explicit function parameter instead of injecting into config dict
- SEC-08: fix fl/__init__.py docstring to state Flower is a required dependency (not optional). No env var interpolation changes needed -- feature doesn't exist in current code

### Claude's Discretion

- Exact wording of validation error messages (as long as they include actual vs expected values)
- How to reconstruct StandardScaler from JSON params (implementation detail)
- Test structure and organization for SEC-09 validation tests
- Whether to add a `FederationError` alongside `DataValidationError` for fedavg-specific failures, or reuse `DataValidationError`

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SEC-01 | Replace all `assert`-based validation with `if/raise ValueError` in loader.py, preprocess.py, partition.py | 10 assert statements located precisely: loader.py (lines 122-134, 3 asserts), preprocess.py (lines 380-397, 6 asserts), partition.py (line 79, 1 assert). Custom `DataValidationError` exception pattern documented. |
| SEC-02 | Add input validation to `fedavg_aggregate` (empty results, zero examples, NaN params) | `fedavg_aggregate` at server.py:46 has zero validation currently. Three validation gates needed: empty list check, zero-example filtering with warning, NaN parameter filtering with warning. |
| SEC-03 | Replace joblib pickle serialization of scaler with safe JSON format | Single save site at preprocess.py:351-352 (`joblib.dump`). No load site exists in source code. Versioned JSON envelope format specified. StandardScaler params: `mean_`, `scale_`, `var_`, `n_features_in_`. |
| SEC-04 | Remove unused `flwr` and `tqdm` from pyproject.toml dependencies | Confirmed: zero imports of `flwr` or `tqdm` anywhere in source. Both are listed in pyproject.toml dependencies (lines 18-19). Safe to remove. |
| SEC-05 | Declare `joblib` as explicit dependency (NOW: remove joblib entirely) | Context decision overrides: since pickle serialization is replaced with JSON (SEC-03), `joblib` is no longer needed at all. Remove the import from preprocess.py:37 and the `joblib.dump` call at line 352. No need to add joblib to pyproject.toml. |
| SEC-06 | Validate `log_level` config and `config_path` inputs before use | 5 entry points use `log_level`: preprocess.py:473, server.py:313, pipeline.py:106, train.py:215, eval/__main__.py:69. All follow identical pattern. Centralize validation in config.py or a shared helper. `config_path` extension validation at load_config entry. |
| SEC-07 | Eliminate config dict mutation side effects (`config["_device"]`) | Mutation at server.py:462. Read at server.py:224 (inside `save_fl_metrics`). Fix: pass `device` as parameter to `save_fl_metrics` instead of injecting into config dict. |
| SEC-08 | Fix Flower dependency contradiction in fl/__init__.py docstring | Current docstring says "no Flower dependency" at line 5. Since `flwr` is being removed from pyproject.toml (SEC-04), the docstring is actually becoming correct. Just clean up the wording to be clear and accurate. |
| SEC-09 | Add tests for validation gates (assert bypass scenario) and fedavg edge cases | Existing test infrastructure: pytest with conftest.py fixtures, class-based tests. New test file `tests/test_security.py` needed for validation gate tests and fedavg edge case tests. |

</phase_requirements>

## Standard Stack

### Core (No New Dependencies)

This phase adds no new libraries. All changes use Python stdlib and existing project dependencies.

| Library | Version | Purpose | Role in Phase 8 |
|---------|---------|---------|-----------------|
| Python stdlib `json` | 3.11+ | JSON serialization | Replace joblib/pickle for scaler save |
| Python stdlib `math` | 3.11+ | `math.isfinite` | NaN/Inf validation in fedavg |
| numpy | >=1.26.0 | Array operations | `np.isnan`, `np.isfinite` for parameter validation |
| scikit-learn | >=1.5.0 | StandardScaler | Reconstruct scaler from JSON params |
| pytest | >=8.0.0 | Testing | SEC-09 validation tests |

### Dependencies to Remove

| Library | Current Location | Reason for Removal |
|---------|-----------------|-------------------|
| `flwr` (>=1.13.0) | pyproject.toml line 18 | Zero imports in source code |
| `tqdm` (>=4.66.0) | pyproject.toml line 19 | Zero imports in source code |
| `joblib` | preprocess.py:37 (import only) | Replaced by JSON serialization (SEC-03) |

## Architecture Patterns

### Pattern 1: Custom Exception Hierarchy

**What:** Single custom exception at package root, subclassing `ValueError`.
**When to use:** All data validation failures across loader, preprocess, partition modules.

```python
# src/federated_ids/exceptions.py
class DataValidationError(ValueError):
    """Raised when input data fails validation checks.

    All messages include actual vs expected values for debugging.
    """
    pass
```

**Rationale for `ValueError` subclass:** Existing callers that catch `ValueError` will still work. The custom type enables targeted `except DataValidationError` handling without breaking existing error handling chains.

### Pattern 2: Assert-to-Raise Replacement

**What:** Replace `assert condition, message` with `if not condition: raise DataValidationError(message)`.
**Why:** `assert` statements are stripped when Python runs with `-O` (optimize) flag, silently disabling all validation gates.

```python
# BEFORE (disabled by python -O):
assert not np.isinf(numeric_df.values).any(), "VALIDATION FAILED: Inf values remain"

# AFTER (always active):
if np.isinf(numeric_df.values).any():
    inf_count = np.isinf(numeric_df.values).sum()
    raise DataValidationError(
        f"Inf values remain after cleaning: found {inf_count} Inf values in numeric columns"
    )
```

**Key principle:** Error messages MUST include actual values (counts, shapes, specific values) for debuggability.

### Pattern 3: Defensive Aggregation with Filtering

**What:** Validate inputs before aggregation, skip invalid clients with warnings, fail only when no valid clients remain.

```python
def fedavg_aggregate(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    # Gate 1: empty results
    if not results:
        raise DataValidationError(
            "fedavg_aggregate received empty results -- no clients participated"
        )

    # Gate 2: filter zero-example clients
    valid_results = []
    for i, (params, n_examples) in enumerate(results):
        if n_examples <= 0:
            logger.warning("Client %d has %d examples, skipping", i, n_examples)
            continue
        # Gate 3: filter NaN parameters
        if any(np.isnan(p).any() for p in params):
            logger.warning("Client %d submitted NaN parameters, skipping", i)
            continue
        valid_results.append((params, n_examples))

    if not valid_results:
        raise DataValidationError(
            "All clients filtered out -- no valid results for aggregation "
            f"(started with {len(results)} clients)"
        )

    # Proceed with valid_results only
    ...
```

### Pattern 4: Versioned JSON Envelope for Scaler

**What:** Serialize StandardScaler parameters as a versioned JSON document instead of pickle.
**Why:** Pickle deserialization executes arbitrary code. JSON is safe, human-readable, and version-trackable.

```python
# Save
def _save_scaler_json(scaler: StandardScaler, path: str) -> None:
    envelope = {
        "version": "1.0",
        "type": "StandardScaler",
        "params": {
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "var_": scaler.var_.tolist(),
            "n_features_in_": int(scaler.n_features_in_),
        },
    }
    with open(path, "w") as f:
        json.dump(envelope, f, indent=2)

# Load (reconstruct)
def _load_scaler_json(path: str) -> StandardScaler:
    with open(path) as f:
        envelope = json.load(f)

    params = envelope["params"]
    n_features = params["n_features_in_"]

    # Validate array lengths
    for key in ("mean_", "scale_", "var_"):
        arr = params[key]
        if len(arr) != n_features:
            raise DataValidationError(
                f"Scaler {key} has {len(arr)} elements, expected {n_features}"
            )

    # Validate all values are finite
    for key in ("mean_", "scale_", "var_"):
        arr = np.array(params[key])
        if not np.all(np.isfinite(arr)):
            raise DataValidationError(
                f"Scaler {key} contains non-finite values"
            )

    # Reconstruct
    scaler = StandardScaler()
    scaler.mean_ = np.array(params["mean_"])
    scaler.scale_ = np.array(params["scale_"])
    scaler.var_ = np.array(params["var_"])
    scaler.n_features_in_ = n_features
    scaler.n_samples_seen_ = 1  # sentinel; exact count not preserved
    return scaler
```

### Pattern 5: Config Validation Centralization

**What:** Add `log_level` and `config_path` validation to the existing `config.py` module.

```python
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

def validate_log_level(level: str) -> str:
    """Validate log_level against allowed values."""
    if level not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log_level '{level}'. Must be one of: {sorted(_VALID_LOG_LEVELS)}"
        )
    return level

def validate_config_path(path: str) -> str:
    """Validate config file has .yaml or .yml extension."""
    if not path.endswith(('.yaml', '.yml')):
        raise ValueError(
            f"Config path '{path}' must have .yaml or .yml extension"
        )
    return path
```

### Pattern 6: Eliminating Config Dict Mutation

**What:** Pass `device` as explicit parameter instead of mutating config dict.

```python
# BEFORE (server.py:462):
config["_device"] = str(device)
metrics_path = os.path.join(output_dir, "metrics", "fl_metrics.json")
save_fl_metrics(history, config, metrics_path)

# AFTER:
metrics_path = os.path.join(output_dir, "metrics", "fl_metrics.json")
save_fl_metrics(history, config, metrics_path, device=str(device))

# save_fl_metrics signature change:
def save_fl_metrics(
    history: list[dict],
    config: dict,
    output_path: str,
    device: str = "cpu",  # NEW parameter
) -> None:
    # ...
    payload = {
        "config": {
            # ...
            "device": device,  # was: str(config.get("_device", "cpu"))
            # ...
        },
        # ...
    }
```

### Anti-Patterns to Avoid

- **Catching DataValidationError and re-raising as ValueError:** The whole point of the custom exception is targeted catching. Don't mask it.
- **Adding try/except around assert replacements:** The raises should propagate. No silent swallowing.
- **Logging before raising:** The CONTEXT.md explicitly forbids this. Just raise with descriptive message.
- **Partial pickle migration (fallback reading):** No pickle fallback. Clean break with clear migration error message.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NaN detection in arrays | Manual loop over array elements | `np.isnan(arr).any()` | Vectorized, handles edge cases |
| Inf detection in arrays | Manual comparison | `np.isinf(arr).any()` or `np.isfinite(arr).all()` | Handles +inf, -inf correctly |
| StandardScaler reconstruction | Manual normalization math | Set `scaler.mean_`, `scaler.scale_`, `scaler.var_`, `scaler.n_features_in_` directly on a fresh `StandardScaler()` instance | Ensures `.transform()` works identically |
| Log level validation | Regex or if-chain | Set membership check against `{"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}` | Complete, readable, maintainable |

## Common Pitfalls

### Pitfall 1: StandardScaler Reconstruction Missing `n_samples_seen_`
**What goes wrong:** After setting `mean_`, `scale_`, `var_`, calling `scaler.partial_fit()` or certain methods may fail because `n_samples_seen_` is not set.
**Why it happens:** `StandardScaler` tracks `n_samples_seen_` during fit. When reconstructing from JSON, this attribute is missing.
**How to avoid:** Set `scaler.n_samples_seen_ = np.int64(1)` (or a reasonable sentinel). The value doesn't affect `transform()` behavior, which only uses `mean_` and `scale_`.
**Warning signs:** `AttributeError: 'StandardScaler' object has no attribute 'n_samples_seen_'` during later operations.

### Pitfall 2: Forgetting to Update Test Assertions for scaler.pkl -> scaler.json
**What goes wrong:** `test_preprocess.py:TestArtifactsSaved` checks for `scaler.pkl` (line 204). After migration, this test will fail.
**Why it happens:** Test references old filename.
**How to avoid:** Update test to check for `scaler.json` instead. Also verify the JSON content structure (version, type, params).

### Pitfall 3: FedAvg Zero Division After Filtering
**What goes wrong:** After filtering out zero-example and NaN clients, `total_examples` could be zero, causing division by zero in weight calculation.
**Why it happens:** All clients filtered out but code continues to aggregation math.
**How to avoid:** Check `if not valid_results` AFTER all filtering, BEFORE any math.

### Pitfall 4: NaN Check Must Cover All Layers
**What goes wrong:** Checking only the first parameter array for NaN misses NaN in later layers.
**Why it happens:** Partial validation.
**How to avoid:** Use `any(np.isnan(p).any() for p in params)` to check ALL parameter arrays.

### Pitfall 5: Config Mutation in Tests
**What goes wrong:** The existing test `TestMetricsJsonOutput` and `TestConfigDrivenRoundsClients` pass config dicts without `_device`. After removing the mutation, `save_fl_metrics` must handle the device parameter explicitly.
**Why it happens:** Tests relied on the mutation happening inside `run_federated_training`.
**How to avoid:** Update `save_fl_metrics` signature to accept `device` as explicit parameter with default `"cpu"`.

### Pitfall 6: Five Entry Points Need Consistent log_level Validation
**What goes wrong:** Adding validation in only some entry points leaves others vulnerable to invalid log levels causing `AttributeError` from `getattr(logging, bad_level)`.
**Why it happens:** The `log_level = config.get("log_level", "INFO")` / `getattr(logging, log_level)` pattern is duplicated in 5 files.
**How to avoid:** Centralize validation. Either: (a) validate in `load_config()` / `_validate_config()` so it happens once, or (b) create a helper `get_validated_log_level(config)` used by all entry points. Option (a) is cleaner since the config is always loaded before use.

### Pitfall 7: fl/__init__.py Docstring Correction Must Match SEC-04
**What goes wrong:** If SEC-04 (remove flwr from deps) is done first but SEC-08 (fix docstring) is not updated, the docstring could contradict reality.
**Why it happens:** Current docstring says "no Flower dependency." After SEC-04 removes flwr from pyproject.toml, this is actually correct. The issue is the old finding from code review was about a previous docstring version.
**How to avoid:** Re-read current docstring carefully: it already says "no Flower dependency" (line 5: "Uses plain NumPy parameter transport (no Flower dependency)"). This is accurate. The SEC-08 requirement was about a contradiction -- verify if there was ever a conflicting statement elsewhere. The current docstring is already correct. Just ensure it clearly states the project does NOT depend on Flower.

## Code Examples

### Assert Replacement Inventory (Exact Locations)

**loader.py** -- 3 asserts (lines 122-134):
```python
# Assert 1 (line 122-124): Inf check
assert not np.isinf(numeric_df.values).any(), "VALIDATION FAILED: Inf values remain after cleaning"

# Assert 2 (line 126-128): NaN check
assert not combined.isna().any().any(), "VALIDATION FAILED: NaN values remain after cleaning"

# Assert 3 (line 130-134): Label check
assert unique_labels == {0, 1}, f"VALIDATION FAILED: Labels are {unique_labels}, expected {{0, 1}}"
```

**preprocess.py** -- 6 asserts (lines 380-397):
```python
# Assert 1 (line 380-382): NaN in X_train
assert not np.isnan(X_train).any(), "VALIDATION FAILED: NaN values in X_train after scaling"
# Assert 2 (line 383-385): NaN in X_test
assert not np.isnan(X_test).any(), "VALIDATION FAILED: NaN values in X_test after scaling"
# Assert 3 (line 386-388): Inf in X_train
assert not np.isinf(X_train).any(), "VALIDATION FAILED: Inf values in X_train after scaling"
# Assert 4 (line 389-391): Inf in X_test
assert not np.isinf(X_test).any(), "VALIDATION FAILED: Inf values in X_test after scaling"
# Assert 5 (line 392-394): X_train dtype
assert X_train.dtype == np.float32, f"VALIDATION FAILED: X_train dtype is {X_train.dtype}, expected float32"
# Assert 6 (line 395-397): X_test dtype
assert X_test.dtype == np.float32, f"VALIDATION FAILED: X_test dtype is {X_test.dtype}, expected float32"
```

**partition.py** -- 1 assert (line 79-82):
```python
# Assert 1 (line 79-82): Class ratio deviation
assert deviation <= 0.05, (
    f"VALIDATION FAILED: Client {fold_idx} class-1 ratio {part_ratio:.3f} "
    f"deviates {deviation:.3f} from global {global_ratio:.3f} (max 0.05)"
)
```

### Config _device Mutation Site

**server.py:462** (mutation):
```python
config["_device"] = str(device)  # THIS LINE MUST BE REMOVED
```

**server.py:224** (read site in `save_fl_metrics`):
```python
"device": str(config.get("_device", "cpu")),  # MUST CHANGE to use explicit parameter
```

### Entry Points Using log_level (5 locations, identical pattern)

All five follow this exact pattern:
```python
log_level = config.get("log_level", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
```

Files: `preprocess.py:473-477`, `server.py:313-317`, `pipeline.py:106-110`, `train.py:215-219`, `eval/__main__.py:69-73`.

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `assert` for validation | `if/raise` with custom exceptions | Validation survives `-O` flag |
| `joblib.dump` (pickle) | JSON serialization | Eliminates arbitrary code execution risk |
| Config dict mutation | Explicit function parameters | No hidden side effects |
| No input validation in fedavg | Pre-aggregation filtering | Graceful degradation with warnings |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0.0 |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/test_security.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SEC-01 | Assert replacement -- Inf raises DataValidationError | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_loader_inf_raises -x` | Wave 0 |
| SEC-01 | Assert replacement -- NaN raises DataValidationError | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_loader_nan_raises -x` | Wave 0 |
| SEC-01 | Assert replacement -- bad labels raises DataValidationError | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_loader_bad_labels_raises -x` | Wave 0 |
| SEC-01 | Assert replacement -- preprocess NaN/Inf/dtype raises | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_preprocess_validation -x` | Wave 0 |
| SEC-01 | Assert replacement -- partition ratio raises | unit | `python -m pytest tests/test_security.py::TestAssertReplacements::test_partition_ratio_raises -x` | Wave 0 |
| SEC-02 | FedAvg rejects empty results | unit | `python -m pytest tests/test_security.py::TestFedAvgValidation::test_empty_results -x` | Wave 0 |
| SEC-02 | FedAvg skips zero-example clients | unit | `python -m pytest tests/test_security.py::TestFedAvgValidation::test_zero_example_clients -x` | Wave 0 |
| SEC-02 | FedAvg skips NaN parameter clients | unit | `python -m pytest tests/test_security.py::TestFedAvgValidation::test_nan_params -x` | Wave 0 |
| SEC-02 | FedAvg fails when ALL clients invalid | unit | `python -m pytest tests/test_security.py::TestFedAvgValidation::test_all_clients_invalid -x` | Wave 0 |
| SEC-03 | Scaler saved as JSON with correct envelope | unit | `python -m pytest tests/test_security.py::TestScalerJson::test_save_json -x` | Wave 0 |
| SEC-03 | Scaler loaded from JSON reconstructs correctly | unit | `python -m pytest tests/test_security.py::TestScalerJson::test_load_json -x` | Wave 0 |
| SEC-03 | Scaler migration error for .pkl only | unit | `python -m pytest tests/test_security.py::TestScalerJson::test_pkl_migration_error -x` | Wave 0 |
| SEC-06 | Invalid log_level raises ValueError | unit | `python -m pytest tests/test_security.py::TestConfigValidation::test_invalid_log_level -x` | Wave 0 |
| SEC-06 | Invalid config_path extension raises ValueError | unit | `python -m pytest tests/test_security.py::TestConfigValidation::test_invalid_config_extension -x` | Wave 0 |
| SEC-07 | Config dict not mutated after FL training | unit | `python -m pytest tests/test_security.py::TestConfigSafety::test_no_config_mutation -x` | Wave 0 |
| SEC-09 | Validation gates catch invalid inputs (integration) | integration | `python -m pytest tests/test_security.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_security.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_security.py` -- new file covering SEC-01 through SEC-09 validation tests
- [ ] `src/federated_ids/exceptions.py` -- new module defining `DataValidationError`
- [ ] Update `tests/test_preprocess.py` line 204 -- change `scaler.pkl` to `scaler.json` artifact check

## Open Questions

1. **Whether to add `FederationError` alongside `DataValidationError`**
   - What we know: CONTEXT.md leaves this to Claude's discretion. `DataValidationError` is for data validation failures. FedAvg validation is arguably about federation protocol violations (no valid clients), not data format errors.
   - Recommendation: Use `DataValidationError` for ALL validation failures in this phase. A separate `FederationError` adds unnecessary complexity for 2-3 error sites. Can be refactored later if the exception hierarchy grows.

2. **Where to add log_level validation -- in `_validate_config` or at each call site**
   - What we know: 5 entry points use the identical pattern. `_validate_config` in config.py already validates section structure.
   - Recommendation: Add `log_level` validation to `_validate_config()` in config.py. This ensures validation happens once at config load time. All 5 entry points already call `load_config()` which calls `_validate_config()`. However, `log_level` is optional (has default "INFO"), so validate only if present. Also add `validate_log_level()` as a public function for direct callers.

3. **Whether `n_samples_seen_` needs exact preservation in scaler JSON**
   - What we know: `n_samples_seen_` is set during `fit()`. It's used by `partial_fit()` but NOT by `transform()`. The project never calls `partial_fit()`.
   - Recommendation: Don't include in JSON envelope. Set to `np.int64(1)` as sentinel on reconstruction. Document this limitation in the JSON format docs.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: all 22 source files and 11 test files examined
- `pyproject.toml` for dependency declarations
- `config.py` for existing validation patterns
- `server.py` for fedavg_aggregate implementation and _device mutation
- `preprocess.py` for scaler serialization and assert locations
- `loader.py` for assert locations
- `partition.py` for assert location

### Secondary (MEDIUM confidence)
- scikit-learn StandardScaler internal attributes (`mean_`, `scale_`, `var_`, `n_features_in_`, `n_samples_seen_`) -- verified from scikit-learn documentation and standard usage patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all changes in existing codebase
- Architecture: HIGH -- patterns are straightforward Python error handling
- Pitfalls: HIGH -- derived from direct code inspection of exact lines
- Test mapping: HIGH -- based on existing test infrastructure and conftest.py patterns

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (stable -- internal refactoring, no external API dependencies)
