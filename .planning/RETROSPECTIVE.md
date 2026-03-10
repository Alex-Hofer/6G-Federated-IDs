# Retrospective

## Milestone: v1.0 — 6G Federated IDS MVP

**Shipped:** 2026-03-10
**Phases:** 7 | **Plans:** 16 | **Timeline:** 2 days

### What Was Built

- CICIDS2017 data pipeline with feature selection, normalization, IID partitioning, and tensor caching
- Configurable MLP model with class-weighted loss and F1-based checkpointing
- FedAvg federated training loop with per-round metrics and convergence verification
- Evaluation suite: confusion matrix, classification report, convergence plots, per-client comparison, TensorBoard
- Single-command pipeline (`federated-ids-run-all`) with thesis-reproducibility README
- Automated verification scripts proving all 17 requirements satisfied

### What Worked

- **Research-first approach:** Phase research identified 10 common pitfalls before coding began, preventing rework
- **Linear dependency chain:** Data -> Model -> FL -> Eval -> Integration built naturally, each phase consumed the previous
- **Tensor caching:** .pt file caching made iteration fast after first run
- **TDD for bug fixes:** Phases 6-7 used failing-test-first approach, caught weighted_loss bug cleanly
- **Verification phases:** Phases 6-7 retroactively verified requirements that would have been orphaned

### What Was Inefficient

- **Verification after the fact:** Phases 6-7 existed solely because verification wasn't built into Phases 1 and 4. Inline verification during implementation would have eliminated two full phases.
- **ROADMAP tracking lag:** Phase 7 was marked "In Progress" after both plans completed — manual status updates are error-prone
- **Undeclared dependencies:** Pillow and joblib not in pyproject.toml, discovered late by audit

### Patterns Established

- YAML config with env var interpolation and validation
- Domain shortlist + statistical feature selection pipeline
- StratifiedKFold for IID client partitioning
- Fresh optimizer per client per FL round (avoids shared state bugs)
- `matplotlib.use('Agg')` before pyplot import for headless environments
- Conditional TensorBoard import with graceful degradation

### Key Lessons

1. **Verify requirements as you build them**, not in cleanup phases afterward
2. **Declare all dependencies** explicitly in pyproject.toml, even transitives
3. **Config flags must be wired** — `fraction_fit` was decorative because the FL loop always iterated all clients
4. **Private symbols (`_cache_exists`) leak** across module boundaries quickly — promote to public API early

### Cost Observations

- Sessions: ~5 sessions across 2 days
- Execution time: ~1 hour for 16 plans (avg 4min/plan)
- Phases 5-7 were fastest (2-3min/plan) due to established patterns

---

## Cross-Milestone Trends

| Metric | v1.0 |
|--------|------|
| Phases | 7 |
| Plans | 16 |
| Avg plan duration | 4min |
| Total time | ~1hr |
| LOC (Python) | 6,643 |
| Requirements | 17/17 |
| Tech debt items | 10 |

---
*Last updated: 2026-03-10*
