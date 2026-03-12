# Phase 8: Security & Integrity - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

All validation is production-safe and dependencies are correctly declared. Replace assert-based validation with proper error handling, add input validation to fedavg_aggregate, fix dependency declarations, replace pickle serialization with safe JSON, validate config inputs, and eliminate config dict mutation. Add tests for validation gates and fedavg edge cases.

Requirements: SEC-01 through SEC-09.

</domain>

<decisions>
## Implementation Decisions

### Error handling style
- Use a custom `DataValidationError` exception (subclass of `ValueError`) for all data validation failures
- Define `DataValidationError` in a new `src/federated_ids/exceptions.py` module at the package root
- Error messages must be detailed with actual vs expected values (e.g., "Expected labels {0, 1}, got {0, 1, 2}" or "Found 42 NaN values in X_train")
- Do NOT log before raising — just raise with a descriptive message; callers handle logging if needed
- Replace all 10 assert statements: loader.py (3), preprocess.py (6), partition.py (1)

### Scaler serialization
- Replace `joblib.dump(scaler, scaler_path)` with JSON serialization
- Use versioned JSON envelope format: `{"version": "1.0", "type": "StandardScaler", "params": {"mean_": [...], "scale_": [...], "var_": [...], "n_features_in_": 78}}`
- File name: `scaler.json` (replaces `scaler.pkl`)
- On load: strict validation — verify array lengths match `n_features_in_`, all values are finite
- Migration: if only `.pkl` exists, raise error with message: "Run `federated-ids-preprocess` to regenerate scaler in safe JSON format" — no auto-conversion, no pickle fallback
- After migration, `joblib` import can be removed from preprocess.py (SEC-05 becomes: don't need to declare joblib at all)

### FedAvg validation strictness
- Empty results list: raise immediately with clear error ("fedavg_aggregate received empty results — no clients participated")
- Zero-example clients: skip with warning log, exclude from aggregation. Error only if ALL clients have zero examples after filtering
- NaN detection: check input parameters BEFORE aggregation (validate each client's submitted weights)
- NaN action: skip client with warning log ("Client X submitted NaN parameters, skipping"), continue with remaining valid clients. Error only if ALL clients are invalid after filtering

### Config safety
- Validate `log_level` against allowlist: {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}. Raise `ValueError` if invalid, showing the bad value and allowed options
- Validate `config_path` extension: must end with `.yaml` or `.yml`. Raise `ValueError` if not (existence already checked by `load_config`)
- Eliminate `config["_device"]` mutation in `server.py:462`: pass `device` as an explicit function parameter instead of injecting into config dict
- SEC-08: fix fl/__init__.py docstring to state Flower is a required dependency (not optional). No env var interpolation changes needed — feature doesn't exist in current code

### Claude's Discretion
- Exact wording of validation error messages (as long as they include actual vs expected values)
- How to reconstruct StandardScaler from JSON params (implementation detail)
- Test structure and organization for SEC-09 validation tests
- Whether to add a `FederationError` alongside `DataValidationError` for fedavg-specific failures, or reuse `DataValidationError`

</decisions>

<specifics>
## Specific Ideas

- User wants versioned envelope for scaler JSON specifically to ensure backward compatibility and strict schema validation for future IDS iterations
- Zero-example and NaN clients should be handled consistently: skip with warning, fail only when no valid clients remain
- "Fail with migration instructions" approach for pickle — clean break, no legacy support

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `config.py` already has `load_config()` with `Path.exists()` check and section type validation — extend for log_level and extension checks
- `tests/` has 11 test files with good patterns (conftest.py, parametrized tests) — SEC-09 tests follow existing conventions

### Established Patterns
- All entry points (`preprocess.py`, `train.py`, `server.py`, `eval/__main__.py`, `pipeline.py`) follow same pattern: `log_level = config.get("log_level", "INFO")` then `getattr(logging, log_level)` — centralize validation in one place
- `fedavg_aggregate` in `server.py:46` currently has no validation at all — called from `server.py:15` and `server.py:415`

### Integration Points
- Assert replacements in `loader.py`, `preprocess.py`, `partition.py` — data pipeline validation layer
- `fedavg_aggregate` validation in `server.py` — FL aggregation layer
- `config["_device"]` mutation at `server.py:462`, read at `server.py:224` — need to trace all usages
- `joblib` import and `joblib.dump()` in `preprocess.py:37,352` — scaler save path
- `pyproject.toml` dependencies: remove unused deps (SEC-04), no longer need to add joblib (SEC-05 resolved by removing pickle)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 08-security-integrity*
*Context gathered: 2026-03-12*
