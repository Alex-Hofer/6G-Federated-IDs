# Phase 5: Integration and Polish - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Tie all components into a single runnable pipeline with orchestration, end-to-end validation, and documentation. Users can run the entire experiment with a single command and reproduce results from scratch. No new ML capabilities — this is integration, validation, and documentation of existing Phases 1-4.

</domain>

<decisions>
## Implementation Decisions

### Pipeline Runner
- Single Python entry point: new console script `federated-ids-run-all` (consistent with existing CLI pattern)
- Chains: preprocess -> FL training -> evaluation in one command
- Full pipeline only — no stage selection flags (individual stages already have their own CLIs)
- Skip preprocessing if cached tensors exist (consistent with FL training behavior)
- Fail fast with clear error message on stage failure (stages depend on each other)

### Output Organization
- Overwrite same outputs/ directory (no timestamped runs)
- Print end-of-pipeline summary: file listing with sizes + key metrics (F1, precision, recall, accuracy)
- One-glance confirmation that everything was produced and how well the model performed

### README & Documentation
- Thesis-reproducibility guide: someone reading the thesis can clone and reproduce the exact experiment
- English language (standard for academic/open-source Python)
- Include troubleshooting section (missing CSVs, CUDA OOM, TensorBoard port conflicts, Python version)
- Include actual output screenshots (confusion matrix, convergence plot) committed to docs/ folder and embedded in README

### End-to-End Validation
- Integration test with synthetic data (pytest, no real CICIDS2017 needed)
- Synthetic CSV mimics full CICIDS2017 column structure (tests the real preprocessing path)
- ~500 rows, enough to partition across clients and complete FL rounds
- Assertions: output files exist + valid structure (JSON parseable, PNGs non-empty, metrics JSON has expected keys)
- No metric thresholds on synthetic data (results are unpredictable on random data)

### Claude's Discretion
- Exact pipeline runner module location and function naming
- README section ordering and formatting
- Synthetic data generation approach (random vs semi-realistic distributions)
- Integration test fixture design (tmp_path vs custom cleanup)
- TensorBoard documentation depth

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Key constraint: keep everything consistent with the existing 4-entry-point CLI pattern and config system.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `federated_ids.config:load_config()`: YAML loader with env var interpolation, validation — reuse for pipeline runner
- `federated_ids.data.preprocess:main(config_path)`: Programmatic entry point — call directly from pipeline runner
- `federated_ids.fl.server:run_federated_training(config)`: Already auto-runs data pipeline if cache missing
- `federated_ids.eval.__main__:main()`: Full evaluation orchestration — call programmatically
- `federated_ids.seed:set_seed()` and `federated_ids.device:get_device()`: Cross-cutting utilities

### Established Patterns
- All entry points accept optional `config_path` parameter (default: `config/default.yaml`)
- CLI via argparse with `--config` flag
- Console scripts defined in pyproject.toml `[project.scripts]`
- Tensor caching in `data/processed/` with `.pt` files
- Output artifacts to `outputs/{checkpoints,metrics,plots,tensorboard}/`

### Integration Points
- Pipeline runner chains existing entry points: `preprocess:main` -> `run_federated_training` -> `eval:main`
- New console script entry in pyproject.toml
- README.md rewrite (existing content covers individual stages)
- New test file in tests/ for integration test

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-integration-and-polish*
*Context gathered: 2026-03-10*
