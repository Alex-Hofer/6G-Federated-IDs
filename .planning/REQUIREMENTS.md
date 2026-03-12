# Requirements: 6G Federated IDS v1.1

**Defined:** 2026-03-12
**Core Value:** Detect DDoS attacks across a federated network of edge nodes without any client ever sharing its raw network traffic data.
**Milestone Goal:** Resolve all 73 findings from comprehensive code review (.full-review/) to achieve zero technical debt.

## v1.1 Requirements

### Security & Integrity

- [x] **SEC-01**: Replace all `assert`-based validation with `if/raise ValueError` in loader.py, preprocess.py, partition.py
- [x] **SEC-02**: Add input validation to `fedavg_aggregate` (empty results, zero examples, NaN params)
- [x] **SEC-03**: Replace joblib pickle serialization of scaler with safe JSON format
- [x] **SEC-04**: Remove unused `flwr` and `tqdm` from pyproject.toml dependencies
- [x] **SEC-05**: Declare `joblib` as explicit dependency in pyproject.toml
- [x] **SEC-06**: Validate `log_level` config and `config_path` inputs before use
- [x] **SEC-07**: Eliminate config dict mutation side effects (`config["_device"]`)
- [x] **SEC-08**: Fix Flower dependency contradiction in `fl/__init__.py` docstring; add env var interpolation allowlist
- [ ] **SEC-09**: Add tests for validation gates (assert bypass scenario) and fedavg edge cases

### Refactoring & Architecture

- [ ] **REFAC-01**: Extract `build_criterion()` factory (eliminate 4x loss construction duplication)
- [ ] **REFAC-02**: Extract `load_cached_tensors()` utility (eliminate 4x tensor loading duplication)
- [ ] **REFAC-03**: Extract auto-run pipeline pattern into shared utility (4x duplication)
- [ ] **REFAC-04**: Deduplicate `evaluate_detailed`/`evaluate` shared inference logic
- [ ] **REFAC-05**: Deduplicate summary table printing between train.py and server.py
- [ ] **REFAC-06**: Pre-allocate model+optimizer outside FL loop (eliminate per-round allocation)
- [ ] **REFAC-07**: Eliminate numpy-tensor round-trip conversions (keep tensors throughout)
- [ ] **REFAC-08**: Standardize parameter loading pattern (`copy_` vs `load_state_dict`)
- [ ] **REFAC-09**: Add cache invalidation mechanism (hash config params, compare on load)
- [ ] **REFAC-10**: Fix design coupling: promote `_cache_exists` to public API, re-export `train_one_epoch`/`evaluate` from `model/__init__.py`
- [ ] **REFAC-11**: Introduce typed configuration (TypedDict or dataclass replacing raw dict)

### Quality & Hygiene

- [ ] **QUAL-01**: Use `torch.inference_mode()` instead of `torch.no_grad()`, `torch.from_numpy()` instead of `torch.tensor()`
- [ ] **QUAL-02**: Use `np.random.default_rng()` instead of legacy `RandomState`
- [ ] **QUAL-03**: Replace `inplace=True` pandas operations; use vectorized `isin()` for label mapping
- [ ] **QUAL-04**: Migrate `os.path` to `pathlib.Path` consistently; fix `os.path.dirname` on bare filename
- [ ] **QUAL-05**: Add `pin_memory=True` and `num_workers` to DataLoaders for GPU performance
- [ ] **QUAL-06**: Remove unused code: `OrderedDict` import, orphaned `scaler.pkl`/`features.json` writes
- [ ] **QUAL-07**: Add `if __name__ == "__main__"` guards to `__main__.py` files; add missing `__all__` exports
- [ ] **QUAL-08**: Fix version string duplication (`__init__.py` vs `pyproject.toml`); use single source of truth
- [ ] **QUAL-09**: Centralize logging: single `setup_logging()` function, remove scattered `basicConfig()` calls
- [ ] **QUAL-10**: Add unit tests for `evaluate_per_client`, `pipeline.py`, cache invalidation, empty DataLoader, malformed CSV
- [ ] **QUAL-11**: Improve test quality: negative tests, content validation, parametrized tests, `monkeypatch` for env vars
- [ ] **QUAL-12**: Document ML security limitations (Byzantine, gradient leakage); add expected baseline results; fix docstring errors
- [ ] **QUAL-13**: Expand ruff rules (add UP, B, SIM, RUF, PT, PD); add mypy configuration

### Final Infrastructure

- [ ] **INFR-01**: Add LICENSE file (MIT or Apache 2.0)
- [ ] **INFR-02**: Design GitHub Actions CI pipeline (lint + test on push)
- [ ] **INFR-03**: Prepare for remote repository (push, fix placeholder `<your-org>` in README)
- [ ] **INFR-04**: Add `.pre-commit-config.yaml` with ruff hooks
- [ ] **INFR-05**: Generate dependency lockfile (`pip freeze > requirements-lock.txt`)
- [ ] **INFR-06**: Add project metadata to pyproject.toml (authors, license, classifiers); add dependency upper bounds
- [ ] **INFR-07**: Add Makefile for common tasks (lint, test, format, run)

## Future Requirements

None — v1.1 is scoped to resolve all existing findings. New features deferred to v2.0.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Byzantine-resilient aggregation (Krum/Trimmed Mean) | Production security — out of scope for thesis proof-of-concept |
| Differential privacy (DP-SGD) | FL-only privacy sufficient for v1.x |
| Secure aggregation protocol | Added complexity not needed for single-machine simulation |
| Parallel client training (ThreadPoolExecutor) | Architecture prep only; actual parallelism deferred to v2.0 |
| Dockerfile/containerization | Design-only in v1.1; full implementation deferred |
| Full mypy strict mode | Configuration added; strict enforcement deferred |
| Performance benchmarks suite | Testing improvements focus on correctness, not perf benchmarks |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SEC-01 | Phase 8 | Complete |
| SEC-02 | Phase 8 | Complete |
| SEC-03 | Phase 8 | Complete |
| SEC-04 | Phase 8 | Complete |
| SEC-05 | Phase 8 | Complete |
| SEC-06 | Phase 8 | Complete |
| SEC-07 | Phase 8 | Complete |
| SEC-08 | Phase 8 | Complete |
| SEC-09 | Phase 8 | Pending |
| REFAC-01 | Phase 9 | Pending |
| REFAC-02 | Phase 9 | Pending |
| REFAC-03 | Phase 9 | Pending |
| REFAC-04 | Phase 9 | Pending |
| REFAC-05 | Phase 9 | Pending |
| REFAC-06 | Phase 9 | Pending |
| REFAC-07 | Phase 9 | Pending |
| REFAC-08 | Phase 9 | Pending |
| REFAC-09 | Phase 9 | Pending |
| REFAC-10 | Phase 9 | Pending |
| REFAC-11 | Phase 9 | Pending |
| QUAL-01 | Phase 10 | Pending |
| QUAL-02 | Phase 10 | Pending |
| QUAL-03 | Phase 10 | Pending |
| QUAL-04 | Phase 10 | Pending |
| QUAL-05 | Phase 10 | Pending |
| QUAL-06 | Phase 10 | Pending |
| QUAL-07 | Phase 10 | Pending |
| QUAL-08 | Phase 10 | Pending |
| QUAL-09 | Phase 10 | Pending |
| QUAL-10 | Phase 10 | Pending |
| QUAL-11 | Phase 10 | Pending |
| QUAL-12 | Phase 10 | Pending |
| QUAL-13 | Phase 10 | Pending |
| INFR-01 | Phase 11 | Pending |
| INFR-02 | Phase 11 | Pending |
| INFR-03 | Phase 11 | Pending |
| INFR-04 | Phase 11 | Pending |
| INFR-05 | Phase 11 | Pending |
| INFR-06 | Phase 11 | Pending |
| INFR-07 | Phase 11 | Pending |

**Coverage:**
- v1.1 requirements: 40 total
- Mapped to phases: 40
- Unmapped: 0 ✓

**Review findings coverage:**
- Total findings from .full-review/: 73
- Addressed by requirements: 73
- Unaddressed: 0 ✓

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-12 after initial definition*
