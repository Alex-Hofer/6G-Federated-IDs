# Roadmap: 6G Federated IDS

## Milestones

- **v1.0 MVP** — Phases 1-7 (shipped 2026-03-10)
- **v1.1 Code Hardening & Security** — Phases 8-11 (in progress)

## Phases

<details>
<summary>v1.0 MVP (Phases 1-7) — SHIPPED 2026-03-10</summary>

- [x] Phase 1: Project Foundation & Data Pipeline (3/3 plans) — completed 2026-03-09
- [x] Phase 2: Model Definition & Local Training (2/2 plans) — completed 2026-03-09
- [x] Phase 3: Federated Learning Infrastructure (2/2 plans) — completed 2026-03-09
- [x] Phase 4: Evaluation & Visualization (2/2 plans) — completed 2026-03-09
- [x] Phase 5: Integration & Polish (3/3 plans) — completed 2026-03-10
- [x] Phase 6: Verify Phase 1 Requirements (2/2 plans) — completed 2026-03-10
- [x] Phase 7: Verify Phase 4 & Integration Fixes (2/2 plans) — completed 2026-03-10

Full details: milestones/v1.0-ROADMAP.md

</details>

### v1.1 Code Hardening & Security (In Progress)

**Milestone Goal:** Resolve all 73 findings from comprehensive code review to achieve zero technical debt.

- [ ] **Phase 8: Security & Integrity** — Replace assert-based validation, fix dependencies, add validation gates
- [ ] **Phase 9: Refactoring & Architecture** — Consolidate duplicated patterns, introduce typed config, fix design coupling
- [ ] **Phase 10: Quality & Hygiene** — Modernize idioms, centralize logging, expand test coverage, configure strict linting
- [ ] **Phase 11: Final Infrastructure** — Add LICENSE, CI pipeline, pre-commit hooks, Makefile, push preparation

## Phase Details

### Phase 8: Security & Integrity
**Goal**: All validation is production-safe and dependencies are correctly declared
**Depends on**: Phase 7 (v1.0 complete)
**Requirements**: SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06, SEC-07, SEC-08, SEC-09
**Success Criteria** (what must be TRUE):
  1. No `assert` statements are used for input validation anywhere in the codebase — all replaced with `if/raise ValueError` (or appropriate exception)
  2. `fedavg_aggregate` rejects empty results, zero-example clients, and NaN parameters with clear error messages
  3. `pyproject.toml` declares exactly the dependencies the project uses — no unused deps listed, no implicit deps missing; scaler serialization uses safe JSON format instead of pickle
  4. Config inputs (`log_level`, `config_path`) are validated before use; config dict is never mutated with side-effect keys like `_device`
  5. Tests verify that validation gates catch invalid inputs (assert-bypass scenarios) and that fedavg edge cases produce correct errors
**Plans:** 3 plans

Plans:
- [ ] 08-01-PLAN.md — Exception infrastructure, assert replacements, scaler JSON migration
- [ ] 08-02-PLAN.md — FedAvg validation, config hardening, dependency cleanup
- [ ] 08-03-PLAN.md — Security validation tests

### Phase 9: Refactoring & Architecture
**Goal**: Duplicated patterns are consolidated and the codebase has a single, typed configuration source
**Depends on**: Phase 8
**Requirements**: REFAC-01, REFAC-02, REFAC-03, REFAC-04, REFAC-05, REFAC-06, REFAC-07, REFAC-08, REFAC-09, REFAC-10, REFAC-11
**Success Criteria** (what must be TRUE):
  1. Loss construction (`build_criterion`), tensor loading (`load_cached_tensors`), auto-run pipeline pattern, and summary table printing each exist in exactly one place — no duplication
  2. `evaluate_detailed` and `evaluate` share a single inference code path; parameter loading uses one consistent pattern throughout
  3. FL training loop reuses pre-allocated model and optimizer across rounds; no unnecessary numpy-to-tensor round-trip conversions remain
  4. Cache invalidation mechanism detects stale cached data when config parameters change (hash-based comparison)
  5. Configuration is typed (dataclass or TypedDict) with IDE autocompletion, validation, and public API for previously private helpers (`_cache_exists`, `train_one_epoch`, `evaluate`)
**Plans**: TBD

### Phase 10: Quality & Hygiene
**Goal**: Code uses modern Python/PyTorch/NumPy idioms, has comprehensive test coverage, and passes strict linting
**Depends on**: Phase 9
**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05, QUAL-06, QUAL-07, QUAL-08, QUAL-09, QUAL-10, QUAL-11, QUAL-12, QUAL-13
**Success Criteria** (what must be TRUE):
  1. All inference uses `torch.inference_mode()`, all tensor creation uses `torch.from_numpy()`, all RNG uses `np.random.default_rng()` — no legacy patterns remain
  2. File paths use `pathlib.Path` consistently (no `os.path`); pandas operations avoid `inplace=True`; vectorized `isin()` replaces manual label mapping
  3. Logging is centralized via a single `setup_logging()` function with no scattered `basicConfig()` calls; version string has a single source of truth
  4. Test suite includes negative tests, parametrized tests, `monkeypatch` for env vars, and coverage for `evaluate_per_client`, `pipeline.py`, cache invalidation, empty DataLoader, and malformed CSV
  5. Ruff config includes UP, B, SIM, RUF, PT, PD rule sets and mypy configuration is present; codebase passes both without errors
**Plans**: TBD

### Phase 11: Final Infrastructure
**Goal**: Project is ready for public repository hosting with CI, licensing, and developer tooling
**Depends on**: Phase 10
**Requirements**: INFR-01, INFR-02, INFR-03, INFR-04, INFR-05, INFR-06, INFR-07
**Success Criteria** (what must be TRUE):
  1. LICENSE file exists at repository root and `pyproject.toml` contains complete project metadata (authors, license field, classifiers, dependency upper bounds)
  2. GitHub Actions CI pipeline runs lint and tests on every push to main and on pull requests
  3. `.pre-commit-config.yaml` with ruff hooks is present and functional; `make lint` and `make format` invoke the same tools
  4. Makefile provides `make lint`, `make test`, `make format`, and `make run` commands that work out of the box
  5. Repository is push-ready: no placeholder values (`<your-org>`) in README, dependency lockfile generated, all metadata complete
**Plans**: TBD

## Progress

**Execution Order:** 8 -> 9 -> 10 -> 11 (linear dependency chain)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Project Foundation & Data Pipeline | v1.0 | 3/3 | Complete | 2026-03-09 |
| 2. Model Definition & Local Training | v1.0 | 2/2 | Complete | 2026-03-09 |
| 3. Federated Learning Infrastructure | v1.0 | 2/2 | Complete | 2026-03-09 |
| 4. Evaluation & Visualization | v1.0 | 2/2 | Complete | 2026-03-09 |
| 5. Integration & Polish | v1.0 | 3/3 | Complete | 2026-03-10 |
| 6. Verify Phase 1 Requirements | v1.0 | 2/2 | Complete | 2026-03-10 |
| 7. Verify Phase 4 & Integration Fixes | v1.0 | 2/2 | Complete | 2026-03-10 |
| 8. Security & Integrity | v1.1 | 2/3 | In progress | - |
| 9. Refactoring & Architecture | v1.1 | 0/TBD | Not started | - |
| 10. Quality & Hygiene | v1.1 | 0/TBD | Not started | - |
| 11. Final Infrastructure | v1.1 | 0/TBD | Not started | - |
