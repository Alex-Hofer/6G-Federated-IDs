# Phase 1: Project Foundation and Data Pipeline - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Scaffold the project and build a validated CICIDS2017 preprocessing pipeline that produces clean, normalized, partitioned PyTorch DataLoaders ready for ML training. All project infrastructure (package, config, seeds, tests) is established here. No model training or federated learning in this phase.

</domain>

<decisions>
## Implementation Decisions

### Data Acquisition
- Manual download — user places CSVs in data/raw/ manually, README documents the UNB source URL and expected files
- Expect original UNB filenames (e.g., 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
- Load only DDoS-relevant CSV files (Friday-WorkingHours), not all 8 CSVs
- Data layout: data/raw/ for original CSVs, data/processed/ for pipeline output, both gitignored

### Feature Selection
- Hybrid approach: start with domain-informed shortlist of DDoS-relevant features (flow duration, packet counts, flag counts, etc.), then apply statistical filters (drop zero-variance, near-constant, highly correlated >0.95)
- Target 20-30 informative features from the 78+ raw columns
- Detailed logging: print/save a report of dropped features with reasons (constant, correlated, etc.) and the final feature list
- Binary label: map all DDoS subtypes (DDoS, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest) to 1, BENIGN to 0

### Data Splitting and Preprocessing
- Global hold-out test set (20%) created BEFORE client partitioning
- Rows with Inf/NaN values are deleted (not imputed)
- StandardScaler fitted ONLY on training set, then applied to test set and all client partitions (no data leakage)
- All numerical features converted to float32 during preprocessing (PyTorch/NVIDIA standard, reduced memory)

### Project Structure
- Subpackage layout: src/federated_ids/ with subpackages data/, model/, fl/, eval/
- Phase 1 focuses on data/ subpackage: loader.py, preprocess.py, partition.py
- pip + pyproject.toml with pinned dependencies, pip install -e . for dev
- Basic unit tests in tests/: verify no NaN/Inf, correct feature count, correct class ratios after split, scaler fitted on train only
- outputs/ directory for plots, checkpoints, logs

### Device Detection
- Auto-detect CUDA (NVIDIA GPU) or MPS (Apple Silicon), fall back to CPU
- Global seed utility fixing numpy, random, torch, and torch.cuda — configurable via config.yaml

### Pipeline Interface
- Python script entry point: python -m federated_ids.data.preprocess + console_scripts entry point
- Override config with --config path/to/config.yaml flag
- Pipeline saves intermediate artifacts: fitted StandardScaler (.pkl), selected feature list (.json), class distribution stats to data/processed/
- Pipeline both saves processed tensors to data/processed/ AND returns DataLoaders in memory — allows skipping preprocessing on subsequent runs

### Configuration
- Flat YAML config with top-level sections: data, model, training, federation
- Environment variable interpolation supported (e.g., ${DATA_DIR:-./data}) for portability across machines/clusters
- Config validation on load: check all required keys are present, fail fast with clear error messages
- Heavily commented config.yaml — each parameter has a one-line explanation (thesis-appendix ready)
- All directory paths (data/raw, data/processed, logs, checkpoints) configurable via YAML, CWD as default base

### Documentation
- Google-style docstrings for all modules and functions (thesis documentation)
- Clear separation between data loading logic and FL client logic

### Claude's Discretion
- Exact statistical thresholds for feature filtering (variance, correlation cutoffs)
- Specific domain features to include in the initial shortlist
- Internal pipeline logging format (print vs logging module)
- Test framework choice (pytest vs unittest)
- Exact console_scripts entry point name

</decisions>

<specifics>
## Specific Ideas

- Config YAML should be usable directly as a reference in the thesis appendix — heavily commented with parameter explanations
- Feature drop report should be detailed enough to cite in methodology section
- Environment variable interpolation enables running on NVIDIA clusters without modifying config files
- float32 conversion is explicitly chosen to reduce memory footprint during training and FL communication

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield project, no existing code

### Established Patterns
- None yet — Phase 1 establishes all patterns

### Integration Points
- DataLoaders produced here are consumed by model training (Phase 2) and FL clients (Phase 3)
- Config system established here is used by all subsequent phases
- Seed utility established here ensures reproducibility across all phases
- data/processed/ artifacts enable faster iteration in later phases (skip preprocessing)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-project-foundation-and-data-pipeline*
*Context gathered: 2026-03-09*
