---
phase: 01-project-foundation-and-data-pipeline
plan: 01
subsystem: infra
tags: [pyproject, yaml, config, seed, device, pytorch, pyyaml]

# Dependency graph
requires:
  - phase: none
    provides: greenfield project
provides:
  - Installable Python package skeleton (federated-ids-6g)
  - YAML config system with env var interpolation and validation
  - Global seed utility for reproducibility (random, numpy, torch, cuda)
  - Device auto-detection (CUDA > MPS > CPU)
  - Test suite with 9 tests covering config, seed, and device
affects: [01-02, 01-03, all subsequent phases]

# Tech tracking
tech-stack:
  added: [torch, pandas, numpy, scikit-learn, matplotlib, seaborn, flwr, tqdm, PyYAML, pytest, ruff]
  patterns: [EnvYamlLoader subclass for env var interpolation, Google-style docstrings, src-layout package structure]

key-files:
  created:
    - pyproject.toml
    - .gitignore
    - config/default.yaml
    - src/federated_ids/__init__.py
    - src/federated_ids/config.py
    - src/federated_ids/seed.py
    - src/federated_ids/device.py
    - src/federated_ids/data/__init__.py
    - src/federated_ids/model/__init__.py
    - src/federated_ids/fl/__init__.py
    - src/federated_ids/eval/__init__.py
    - tests/conftest.py
    - tests/test_config.py
  modified: []

key-decisions:
  - "EnvYamlLoader subclass of SafeLoader to avoid global YAML loader mutation (Pitfall 7)"
  - "Anchored .gitignore patterns (/data/, /outputs/) to avoid ignoring src/federated_ids/data/"
  - "Config validation checks both top-level sections and required nested keys with descriptive error messages"

patterns-established:
  - "Custom YAML loader: subclass SafeLoader, never modify global loader class"
  - "Google-style docstrings on all modules and public functions"
  - "src-layout package: src/federated_ids/ with subpackages data/, model/, fl/, eval/"
  - "Config-driven design: all hyperparameters in config/default.yaml, never hardcoded"
  - "Environment variable interpolation via ${VAR:-default} syntax in YAML"

requirements-completed: [INFR-01, INFR-02]

# Metrics
duration: 12min
completed: 2026-03-09
---

# Phase 1 Plan 01: Project Scaffold and Infrastructure Summary

**Installable Python package with YAML config system (env var interpolation + validation), global seed utility, and device auto-detection**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-09T13:15:07Z
- **Completed:** 2026-03-09T13:27:17Z
- **Tasks:** 2
- **Files modified:** 13

## Accomplishments
- Created complete Python package skeleton with pyproject.toml, pinned dependencies, and 4 subpackages (data, model, fl, eval)
- Built config loading system with custom EnvYamlLoader that resolves ${VAR:-default} patterns without mutating the global YAML SafeLoader
- Implemented set_global_seed() seeding random, numpy, torch, and cuda for full reproducibility
- Implemented get_device() with CUDA > MPS > CPU priority detection
- Created test suite with 9 tests covering config loading, env var interpolation, validation errors, seed reproducibility, and device detection

## Task Commits

Each task was committed atomically:

1. **Task 1: Create project scaffold with pyproject.toml, package structure, and .gitignore** - `3d476d9` (feat)
2. **Task 2: Implement config loader, seed utility, device detection, and their tests** - `6564451` (feat)

## Files Created/Modified
- `pyproject.toml` - Project metadata, pinned dependencies (torch, pandas, sklearn, flwr, etc.), tool config (ruff, pytest)
- `.gitignore` - Covers /data/, /outputs/, .venv/, __pycache__/, build artifacts, model files
- `config/default.yaml` - Heavily commented config with all hyperparameter sections (data, model, training, federation, seed)
- `src/federated_ids/__init__.py` - Package init with __version__ = "0.1.0"
- `src/federated_ids/config.py` - YAML loading with env var interpolation and validation
- `src/federated_ids/seed.py` - Global seed utility (random, numpy, torch, cuda, cudnn)
- `src/federated_ids/device.py` - Device auto-detection (CUDA > MPS > CPU)
- `src/federated_ids/data/__init__.py` - Data subpackage init with docstring
- `src/federated_ids/model/__init__.py` - Model subpackage init with docstring
- `src/federated_ids/fl/__init__.py` - FL subpackage init with docstring
- `src/federated_ids/eval/__init__.py` - Eval subpackage init with docstring
- `tests/conftest.py` - Shared fixtures (sample_config_dict, tmp_config_file)
- `tests/test_config.py` - 9 tests across 5 test classes

## Decisions Made
- Used EnvYamlLoader subclass instead of modifying global yaml.SafeLoader (Pitfall 7 from research)
- Anchored .gitignore data/ pattern to /data/ to avoid ignoring src/federated_ids/data/ subpackage
- Config validation is two-tier: checks top-level sections exist with correct types, then checks required nested keys within each section

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed .gitignore pattern blocking src/federated_ids/data/**
- **Found during:** Task 1 (git add)
- **Issue:** `data/` pattern in .gitignore matched `src/federated_ids/data/` directory, preventing git tracking of the data subpackage
- **Fix:** Changed to `/data/` (anchored to project root) and `/outputs/` to only ignore top-level directories
- **Files modified:** .gitignore
- **Verification:** git add succeeded for src/federated_ids/data/__init__.py
- **Committed in:** 3d476d9 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix -- without it, the data subpackage would not be tracked by git.

## Issues Encountered
- Python is not installed on this system (only Windows Store stub in WindowsApps). TDD RED-GREEN-REFACTOR cycle could not be executed. Tests are written but not yet run. User must install Python 3.11+ and run `pip install -e ".[dev]" && python -m pytest tests/ -x -v` to verify.

## User Setup Required

None - no external service configuration required. However, Python 3.11+ must be installed before running the project:
```bash
# After installing Python 3.11+
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -e ".[dev]"
python -m pytest tests/ -x -v
```

## Next Phase Readiness
- Package skeleton is complete and ready for Plan 01-02 (data loading, cleaning, feature selection)
- Config system is ready to be consumed by all subsequent modules
- Seed and device utilities are ready for model training (Phase 2)
- Python installation needed before tests can be verified

---
*Phase: 01-project-foundation-and-data-pipeline*
*Completed: 2026-03-09*
