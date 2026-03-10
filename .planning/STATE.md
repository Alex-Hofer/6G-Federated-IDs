---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: complete
stopped_at: "Completed 05-02-PLAN.md"
last_updated: "2026-03-10T09:24:58Z"
last_activity: 2026-03-10 -- Plan 05-02 executed, README rewrite as thesis-reproducibility guide
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 11
  completed_plans: 11
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-09)

**Core value:** Detect DDoS attacks across a federated network of edge nodes without any client ever sharing its raw network traffic data.
**Current focus:** All phases complete. Milestone v1.0 delivered.

## Current Position

Phase: 5 of 5 (Integration & Polish)
Plan: 2 of 2 in current phase (all complete)
Status: Milestone complete
Last activity: 2026-03-10 -- Plan 05-02 executed, README rewrite as thesis-reproducibility guide

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: 5min
- Total execution time: 0.83 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 3 | 20min | 7min |
| 2 | 2 | 6min | 3min |
| 3 | 2 | 11min | 6min |
| 4 | 2 | ~6min | ~3min |
| 5 | 2/2 | 8min | 4min |

**Recent Trend:**
- Last 5 plans: 03-02 (5min), 04-01 (~3min), 04-02 (3min), 05-01 (3min), 05-02 (5min)
- Trend: stable

*Updated after each plan completion*
| Phase 05 P01 | 3min | 2 tasks | 3 files |
| Phase 05 P02 | 5min | 2 tasks | 1 file |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 5-phase linear dependency chain (Data -> Model -> FL -> Eval -> Integration) based on research finding that 7/10 pitfalls originate in data layer
- [Roadmap]: INFR-01 and INFR-02 placed in Phase 1 (not a separate phase) since config and reproducibility are foundational cross-cutting concerns
- [Roadmap]: DATA-04 mapped to both Phase 1 (weight computation) and Phase 2 (weighted loss in training loop) -- primary assignment Phase 1, consumed in Phase 2
- [Roadmap]: EVAL-01 (per-round console metrics) placed in Phase 3 (FL Infrastructure) rather than Phase 4 because per-round logging is integral to FL training, not post-hoc evaluation
- [01-01]: EnvYamlLoader subclass of SafeLoader to avoid global YAML loader mutation (Pitfall 7)
- [01-01]: Anchored .gitignore patterns (/data/, /outputs/) to avoid ignoring src/federated_ids/data/
- [01-01]: Config validation checks both top-level sections and required nested keys with descriptive error messages
- [01-02]: Domain shortlist of 44 DDoS-relevant features with fallback to all numeric columns if fewer than target remain
- [01-02]: Near-constant filter at >99% same value threshold in addition to zero-variance filter
- [01-02]: Correlation filtering keeps the feature with higher variance from each correlated pair
- [01-02]: Class weights saved as JSON (portable) alongside scaler saved as joblib pkl
- [01-03]: StratifiedKFold test indices used as client shards (non-overlapping, cover all data, preserve class ratios)
- [01-03]: Tensor caching with .pt files enables pipeline to skip expensive CSV loading and preprocessing on subsequent runs
- [01-03]: Pipeline entry point supports both CLI (argparse) and programmatic (config_path parameter) invocation
- [02-01]: Raw logits output (no softmax) to avoid double-softmax bug with CrossEntropyLoss
- [02-01]: Dynamic layer construction via nn.Sequential from hidden_layers list
- [02-02]: Class-weighted loss loaded from class_weights.json with explicit device placement
- [02-02]: Best-model checkpoint saved only on F1 improvement (not loss) for better DDoS detection quality
- [02-02]: standalone_epochs and val_split as optional config keys (not in _REQUIRED_NESTED) to avoid breaking FL configs
- [02-02]: Auto-run data pipeline if cached tensors missing for zero-config standalone training
- [03-01]: In-place parameter copy via copy_() instead of load_state_dict to preserve optimizer tensor references
- [03-01]: get_parameters returns numpy copies (not views) to prevent mutation bugs across FL rounds
- [03-02]: Fresh MLP and Adam optimizer created per client per round to avoid shared state bugs (Pitfall 6 & 2)
- [03-02]: Convergence check uses mean F1 of first n vs last n rounds with adaptive n for short histories
- [03-02]: Metrics JSON embeds full config snapshot for thesis reproducibility
- [03-02]: CLI overrides (--num-clients, --num-rounds) mutate config dict before passing to run_federated_training
- [04-01]: evaluate_detailed mirrors train.py:evaluate logic without modifying original (Phase 3 stability)
- [04-01]: Total-based confusion matrix percentages (not row-based) per research Pitfall 2
- [04-01]: Per-client local training uses num_rounds * local_epochs total epochs for fair comparison (Pitfall 3)
- [04-01]: matplotlib.use('Agg') set before pyplot import for headless rendering
- [04-02]: Conditional import with _HAS_TENSORBOARD flag for graceful degradation when tensorboard not installed
- [04-02]: try/finally block wrapping entire FL loop and post-training to guarantee writer.close()
- [04-02]: Global-only metrics (no per-client TB logging) per user decision
- [05-01]: Lazy imports for stage modules to keep pipeline module import lightweight
- [05-01]: No try/except around stages -- fail-fast by letting exceptions propagate
- [05-01]: Registered pytest slow marker in pyproject.toml to suppress unknown mark warning
- [05-02]: README written in English per user decision
- [05-02]: Screenshot embeds use docs/ folder with instructions to generate from pipeline output
- [05-02]: README structured as 15-section thesis-reproducibility guide covering clone-to-results workflow

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1]: CICIDS2017 download URL needs verification -- multiple mirrors exist with different file sets (flagged by research)
- [Phase 3]: Flower API version (1.x classic vs 2.x) must be confirmed during Phase 1 dependency pinning (flagged by research)

## Session Continuity

Last session: 2026-03-10T09:24:58Z
Stopped at: Completed 05-02-PLAN.md -- All plans complete, milestone v1.0 delivered
Resume file: .planning/phases/05-integration-and-polish/05-02-SUMMARY.md
