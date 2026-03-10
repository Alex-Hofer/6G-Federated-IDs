# Phase 7: Verify Phase 4 & Integration Fixes - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Formally verify all Phase 4 requirements (EVAL-02, EVAL-03, EVAL-04) by running code paths with synthetic data, validating outputs, and creating VERIFICATION.md. Fix the standalone_train weighted_loss config inconsistency. No new features — purely verification, bug fix, and gap closure.

</domain>

<decisions>
## Implementation Decisions

### TensorBoard verification (EVAL-04)
- Run a mini FL training (2 clients, 2 rounds, tiny synthetic data) with TB enabled
- Verify that a tfevents file exists in the runs/ directory AND contains expected scalar tags (accuracy, f1, loss)
- tensorboard is a required project dependency — if not installed, the check FAILS (not skipped)
- Clean up temporary TB event files after verification — evidence is captured in VERIFICATION.md

### Plot & report verification (EVAL-02, EVAL-03)
- EVAL-02: Run eval module with synthetic model + synthetic test data, verify confusion matrix PNG and classification report text are produced with expected structure
- EVAL-03: Check convergence PNGs exist, have non-zero file size, and can be opened by PIL/matplotlib as valid images (not corrupt)
- Per-client comparison visualization also verified (part of Phase 4 success criteria)
- All verification outputs generated in a temp directory, checked, then cleaned up
- Evidence captured in VERIFICATION.md, not left as files in the workspace

### weighted_loss fix
- Fix standalone_train only — FL server and eval already check the flag correctly
- When weighted_loss is false (or absent), use plain CrossEntropyLoss() without weights — consistent with FL server and eval behavior
- Add a pytest regression test that verifies standalone_train respects weighted_loss=true and weighted_loss=false
- Update config/default.yaml to explicitly include weighted_loss: true so users see the option

### Verification format
- Follow Phase 6 pattern: standalone script at scripts/verify_phase4.py
- Pass/fail with actual measured values per requirement
- VERIFICATION.md with traceability table in phase directory, thesis-appendix-ready
- Script is re-runnable and produces human-readable output

### Claude's Discretion
- Internal structure of the verification script
- How synthetic data and model are generated for eval verification
- Exact format of human-readable output
- How to parse/check TB event file scalar tags

</decisions>

<specifics>
## Specific Ideas

- VERIFICATION.md should match Phase 6 format for cross-phase consistency
- Verification script should be self-contained — generates its own synthetic data inline (like Phase 6)
- The weighted_loss fix should mirror the exact pattern used in server.py (lines 353-354) and evaluate.py (lines 181-182)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/evaluate.py`: evaluate_detailed function generates confusion matrix, classification report, convergence plots, per-client comparison
- `eval/plots.py`: Plot generation functions for convergence curves and per-client bars
- `eval/__main__.py`: CLI entry point that wires eval together — can be called programmatically
- `fl/server.py`: FL server with TB integration (lines 353-354 show correct weighted_loss pattern)
- `model/train.py`: standalone_train function (lines 323-333 — the bug location)
- `scripts/verify_phase1.py`: Phase 6 verification script — pattern to follow

### Established Patterns
- Conditional TB import with `_HAS_TENSORBOARD` flag in server.py
- `training_config.get("weighted_loss", False)` is the standard pattern for checking the flag
- Verification scripts generate synthetic data inline, don't depend on conftest.py fixtures
- matplotlib.use('Agg') before pyplot import for headless rendering

### Integration Points
- Verification script imports from `federated_ids.eval`, `federated_ids.fl`, `federated_ids.model`
- VERIFICATION.md placed in `.planning/phases/07-verify-phase4-and-integration-fixes/`
- weighted_loss fix touches `model/train.py` and `config/default.yaml`
- Regression test added to existing test suite

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-verify-phase4-and-integration-fixes*
*Context gathered: 2026-03-10*
