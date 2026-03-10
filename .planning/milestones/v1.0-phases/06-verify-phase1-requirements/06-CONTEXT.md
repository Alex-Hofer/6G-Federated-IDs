# Phase 6: Verify Phase 1 Requirements - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Formally verify all Phase 1 requirements (DATA-01, DATA-02, DATA-03, DATA-05, INFR-01) by running code paths, validating outputs, and creating VERIFICATION.md. Fix the missing `select_features` re-export from `data/__init__.py`. No new features — purely verification and gap closure.

</domain>

<decisions>
## Implementation Decisions

### Verification evidence
- Pass/fail with actual measured values per requirement (e.g. "Feature count: 32, range 20-40: PASS")
- Traceability table in VERIFICATION.md mapping each requirement ID to its check, result, and value
- Brief method line per check documenting how it was verified (e.g. "Verified via: scripts/verify_phase1.py")
- Record verification date and environment (Python version, OS) in VERIFICATION.md header

### Test data source
- Both synthetic and real CICIDS2017 data
- Synthetic tests verify code paths and logic
- Real CICIDS2017 run captures actual numeric results (row counts, feature counts, class distributions) for thesis-grade evidence
- Graceful fallback: if CICIDS2017 download fails/unavailable, real data section marked "SKIPPED — data not available" and phase still passes on synthetic results alone
- CICIDS2017 needs to be downloaded first (not yet available locally)

### Verification approach
- Standalone verification script at `scripts/verify_phase1.py`
- Script does its own requirement-specific checks (load data, count features, check scaler, etc.)
- Human-readable output to stdout, captured into VERIFICATION.md
- Existing pytest tests run separately — script does not invoke pytest
- Script must handle missing real data gracefully (fallback to synthetic-only)

### select_features fix
- Add `select_features` re-export to `data/__init__.py` and update `__all__`
- Add a test that imports `select_features` from `federated_ids.data` to prevent future regressions
- Verification script calls `select_features` standalone on a small DataFrame to prove the re-export is functional
- Claude's discretion: check preprocess.py for other useful public functions that should also be re-exported

### Claude's Discretion
- Which other public functions from preprocess.py (if any) to add to data/__init__.py exports
- Internal structure of the verification script (how checks are organized)
- Exact format of the human-readable output
- How to generate/download synthetic test data within the script

</decisions>

<specifics>
## Specific Ideas

- VERIFICATION.md should be thesis-appendix-ready — someone reviewing the thesis should see clear evidence that each requirement was met
- Traceability table format: | REQ ID | Check | Method | Result | Value |
- The script should be re-runnable: `python scripts/verify_phase1.py` produces the same verification report each time

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `data/__init__.py`: Currently exports `load_cicids2017`, `preprocess`, `run_pipeline`, `partition_iid`, `create_dataloaders` — missing `select_features`
- `data/preprocess.py`: Contains `select_features` function and `preprocess` pipeline
- `data/loader.py`: CSV loading and cleaning logic (DATA-01)
- `data/partition.py`: IID partitioning and DataLoader creation (DATA-05)
- `config.py`: YAML config loading with env var expansion (INFR-01)
- Existing tests: `test_config.py`, `test_loader.py`, `test_preprocess.py`, `test_partition.py`

### Established Patterns
- Tests use pytest with synthetic data fixtures
- Pipeline produces cached .pt tensor files in processed_dir
- Config loaded via custom EnvYamlLoader (SafeLoader subclass)
- Domain shortlist of 44 DDoS-relevant features with statistical filtering fallback

### Integration Points
- Verification script imports from `federated_ids.data` and `federated_ids.config`
- VERIFICATION.md placed in `.planning/phases/06-verify-phase1-requirements/`
- `config/default.yaml` is the reference config for INFR-01 verification

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-verify-phase1-requirements*
*Context gathered: 2026-03-10*
