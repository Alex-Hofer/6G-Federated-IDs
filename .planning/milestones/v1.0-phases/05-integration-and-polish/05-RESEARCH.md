# Phase 5: Integration and Polish - Research

**Researched:** 2026-03-10
**Domain:** Pipeline orchestration, end-to-end integration testing, project documentation
**Confidence:** HIGH

## Summary

Phase 5 is an integration-only phase that ties together the four completed phases (Data, Model, FL, Evaluation) into a single runnable pipeline with end-to-end validation and thesis-reproducibility documentation. No new ML capabilities are introduced. The codebase is well-structured with consistent patterns: each stage has a `main()` function accepting `config_path`, all stages auto-run dependencies when cached data is missing, and all entry points are registered as console scripts in pyproject.toml.

The primary work is: (1) a new `pipeline.py` module that chains `preprocess:main` -> `run_federated_training` -> `eval.__main__:main` with a summary printer, (2) an integration test using synthetic CSV data that exercises the full pipeline without requiring real CICIDS2017 files, and (3) a comprehensive README rewrite covering setup through reproduction. The existing codebase provides all the building blocks -- this phase simply wires them together and documents the result.

**Primary recommendation:** Build a thin orchestration layer that calls existing programmatic entry points sequentially, adds a summary printer, and wraps the whole flow in an integration test with synthetic data matching full CICIDS2017 column structure.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Single Python entry point: new console script `federated-ids-run-all` (consistent with existing CLI pattern)
- Chains: preprocess -> FL training -> evaluation in one command
- Full pipeline only -- no stage selection flags (individual stages already have their own CLIs)
- Skip preprocessing if cached tensors exist (consistent with FL training behavior)
- Fail fast with clear error message on stage failure (stages depend on each other)
- Overwrite same outputs/ directory (no timestamped runs)
- Print end-of-pipeline summary: file listing with sizes + key metrics (F1, precision, recall, accuracy)
- Thesis-reproducibility guide: someone reading the thesis can clone and reproduce the exact experiment
- English language (standard for academic/open-source Python)
- Include troubleshooting section (missing CSVs, CUDA OOM, TensorBoard port conflicts, Python version)
- Include actual output screenshots (confusion matrix, convergence plot) committed to docs/ folder and embedded in README
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

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODL-03 | Model checkpointing to save best-performing global model based on F1-score | Already implemented in fl/server.py (saves on F1 improvement). Pipeline runner validates checkpoint exists in summary. Integration test asserts `global_model.pt` is produced. |
| INFR-02 | Reproducibility via fixed seeds, pyproject.toml, and documented hyperparameters | Already implemented via seed.py, pyproject.toml dependencies, and config/default.yaml. Pipeline runner calls `set_global_seed()`. README documents full reproduction steps including exact config. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | >=3.11 | Runtime | Already pinned in pyproject.toml |
| pytest | >=8.0.0 | Integration testing | Already in dev dependencies, used by all 88 existing tests |
| PyYAML | >=6.0 | Config loading | Already a dependency, used by config.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | >=2.2.0 | Synthetic CSV generation for integration test | Creating realistic CICIDS2017 CSV fixtures |
| numpy | >=1.26.0 | Random data generation for synthetic fixtures | Integration test data generation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytest tmp_path | tempfile.TemporaryDirectory | tmp_path is pytest-native, auto-cleaned, provides Path object -- preferred |
| subprocess calls to chain stages | Direct Python function calls | Function calls are faster, share the same process, provide better error tracebacks -- preferred |

**Installation:** No new dependencies needed. Everything is already in pyproject.toml.

## Architecture Patterns

### Recommended Project Structure
```
src/federated_ids/
    pipeline.py            # NEW: run_pipeline() + main() entry point
    ...existing modules unchanged...

tests/
    test_integration.py    # NEW: end-to-end pipeline test with synthetic data
    conftest.py            # EXTEND: add integration test fixtures

docs/
    confusion_matrix.png   # NEW: screenshot for README embedding
    convergence.png        # NEW: screenshot for README embedding

README.md                  # REWRITE: thesis-reproducibility guide
```

### Pattern 1: Thin Orchestrator Calling Existing Entry Points
**What:** Pipeline runner imports and calls existing `main()` functions directly rather than spawning subprocesses or duplicating logic.
**When to use:** When all stages are Python modules in the same package with programmatic entry points.
**Example:**
```python
from federated_ids.data.preprocess import main as run_preprocess, _cache_exists
from federated_ids.fl.server import run_federated_training
from federated_ids.eval.__main__ import main as run_evaluation
from federated_ids.config import load_config
from federated_ids.seed import set_global_seed

def run_pipeline(config_path: str = "config/default.yaml") -> None:
    config = load_config(config_path)
    set_global_seed(config.get("seed", 42))

    # Stage 1: Preprocess (skip if cached)
    processed_dir = config["data"].get("processed_dir", "./data/processed")
    if not _cache_exists(processed_dir):
        run_preprocess(config_path)

    # Stage 2: Federated training
    run_federated_training(config, config_path=config_path)

    # Stage 3: Evaluation
    run_evaluation(config_path)

    # Stage 4: Print summary
    print_summary(config)
```

### Pattern 2: Integration Test with Synthetic CSV
**What:** Generate a full CSV matching CICIDS2017 column structure, write to tmp_path, point config at it, run the entire pipeline, assert output files exist and are valid.
**When to use:** End-to-end validation without requiring real dataset download.
**Example:**
```python
def test_full_pipeline(tmp_path):
    # 1. Generate synthetic CSV with _CICIDS_COLUMNS + Label
    # 2. Write config.yaml pointing raw_dir/processed_dir/output_dir to tmp_path
    # 3. Call run_pipeline(config_path)
    # 4. Assert: global_model.pt exists, fl_metrics.json is valid JSON,
    #    PNGs are non-empty, classification_report.txt exists
```

### Pattern 3: Summary Printer with File Listing
**What:** After all stages complete, walk the output directory, print each file with human-readable size, then print key metrics from fl_metrics.json.
**When to use:** End-of-pipeline confirmation that all artifacts were produced.
**Example:**
```python
def print_summary(config: dict) -> None:
    output_dir = config.get("output_dir", "./outputs")
    # Walk output_dir, list all files with sizes
    # Parse fl_metrics.json for final-round metrics
    # Print F1, precision, recall, accuracy from final round
```

### Anti-Patterns to Avoid
- **Subprocess spawning for stage chaining:** Don't use `subprocess.run(["federated-ids-preprocess"])`. This loses stack traces, creates process overhead, and makes error handling harder. Call the Python functions directly.
- **Duplicating auto-run logic in pipeline runner:** The FL server and eval module already auto-run preprocessing if cached data is missing. The pipeline runner should call preprocessing explicitly (for clarity) but should NOT add redundant cache-checking inside the FL/eval calls since those calls will skip it if cache already exists.
- **Metric threshold assertions in integration test on synthetic data:** Random data produces unpredictable metrics. Assert structure (file exists, JSON valid, expected keys present) not values (F1 > 0.8).
- **Modifying existing modules:** Phase 5 should NOT change any existing Phase 1-4 code. All integration is through existing public APIs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Argument parsing | Custom string parsing | argparse (consistent with all other entry points) | Pattern already established by 4 existing CLIs |
| Synthetic CICIDS2017 data | New column list from scratch | Reuse `_CICIDS_COLUMNS` from conftest.py | Already defines all 78 columns with correct whitespace quirks |
| File size formatting | Manual byte conversion | Simple helper or f-string with / 1024 | Only used once in summary printer |
| Config validation | Custom checks in pipeline runner | Existing `load_config()` already validates | config.py handles all validation centrally |

**Key insight:** The entire point of Phase 5 is that everything already exists. The pipeline runner is a thin wrapper (~50-80 lines) that chains existing functions. Resist the urge to add new functionality.

## Common Pitfalls

### Pitfall 1: Argparse Conflicts When Chaining Entry Points
**What goes wrong:** If `run_preprocess(None)` is called inside the pipeline runner, it creates its own argparse parser and tries to parse sys.argv, which may contain `--config` from the pipeline runner's own parser.
**Why it happens:** Each stage's `main()` defaults to CLI parsing when `config_path is None`.
**How to avoid:** Always pass `config_path` explicitly to every stage call. Never pass `None`.
**Warning signs:** "unrecognized arguments" errors when running the pipeline.

### Pitfall 2: Logging Configuration Clobbering
**What goes wrong:** Each stage calls `logging.basicConfig()` which only takes effect on the FIRST call. Later stages' logging configuration is silently ignored.
**Why it happens:** `basicConfig()` is a no-op if the root logger already has handlers.
**How to avoid:** Call `logging.basicConfig()` once at the start of the pipeline runner. Since existing stages call it too but it's a no-op after the first call, this is safe. The pipeline runner sets it first, and subsequent calls in stages are harmless.
**Warning signs:** Missing log output from later stages.

### Pitfall 3: Working Directory Assumptions in Config Paths
**What goes wrong:** Config uses relative paths like `"./data/raw"` which resolve relative to CWD, not relative to the config file or project root.
**Why it happens:** The existing codebase consistently uses CWD-relative paths (this is by design).
**How to avoid:** Document in README that commands must be run from the project root directory. The pipeline runner should NOT change CWD or rewrite paths.
**Warning signs:** FileNotFoundError for data/config files when running from a different directory.

### Pitfall 4: Integration Test Timeout
**What goes wrong:** The integration test runs the full pipeline including FL training which, even with 2 clients and 2 rounds on synthetic data, takes time for model creation, forward/backward passes, evaluation, and plot generation.
**Why it happens:** Each FL round creates fresh models, runs training, runs evaluation.
**How to avoid:** Use minimal config: 2 clients, 2 rounds, 1 local epoch, small hidden layers [16, 8], batch_size=32, ~500 rows of synthetic data. This should complete in under 30 seconds.
**Warning signs:** Test suite takes minutes instead of seconds.

### Pitfall 5: Synthetic CSV Missing Required Structure
**What goes wrong:** The integration test generates a CSV but misses some columns or label format that the real preprocessing path expects, causing the pipeline to fail in ways that don't happen with real data.
**Why it happens:** CICIDS2017 has 78+ columns with leading whitespace quirks, inf/NaN values, and specific label strings.
**How to avoid:** Reuse the `_CICIDS_COLUMNS` list from `conftest.py` which already includes leading whitespace. Include both "BENIGN" and "DDoS" labels. Inject at least some inf/NaN values to exercise the cleaning path.
**Warning signs:** Pipeline works with real data but fails in integration test.

### Pitfall 6: Plot Generation Fails in Headless Environment
**What goes wrong:** matplotlib tries to open a display.
**Why it happens:** Missing `matplotlib.use('Agg')` call.
**How to avoid:** Already handled -- `plots.py` sets `matplotlib.use("Agg")` at module level. No additional action needed in the pipeline runner.
**Warning signs:** `_tkinter.TclError: no display name` in CI or headless environments.

## Code Examples

### Pipeline Runner Module Structure
```python
# src/federated_ids/pipeline.py
"""End-to-end pipeline runner: preprocess -> FL training -> evaluation.

Chains all stages of the federated IDS experiment in a single command.
Individual stages can still be run independently via their own CLIs.

Usage::

    federated-ids-run-all
    federated-ids-run-all --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from federated_ids.config import load_config
from federated_ids.data.preprocess import _cache_exists
from federated_ids.seed import set_global_seed

logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "config/default.yaml") -> None:
    """Run the complete federated IDS pipeline end-to-end."""
    config = load_config(config_path)
    seed = config.get("seed", 42)
    set_global_seed(seed)

    log_level = config.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    processed_dir = config["data"].get("processed_dir", "./data/processed")
    output_dir = config.get("output_dir", "./outputs")

    # Stage 1: Preprocess
    logger.info("=== Stage 1/3: Data Preprocessing ===")
    if _cache_exists(processed_dir):
        logger.info("Cached tensors found in %s -- skipping preprocessing.", processed_dir)
    else:
        from federated_ids.data.preprocess import main as run_preprocess
        run_preprocess(config_path)

    # Stage 2: Federated Training
    logger.info("=== Stage 2/3: Federated Training ===")
    from federated_ids.fl.server import run_federated_training
    run_federated_training(config, config_path=config_path)

    # Stage 3: Evaluation
    logger.info("=== Stage 3/3: Evaluation ===")
    from federated_ids.eval.__main__ import main as run_evaluation
    run_evaluation(config_path)

    # Summary
    _print_pipeline_summary(output_dir)


def _print_pipeline_summary(output_dir: str) -> None:
    """Print file listing with sizes and key metrics."""
    # ... walk output_dir, list files with sizes ...
    # ... parse fl_metrics.json for final metrics ...


def main() -> None:
    """CLI entry point for federated-ids-run-all."""
    parser = argparse.ArgumentParser(
        description="Run the complete federated IDS pipeline end-to-end."
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    args = parser.parse_args()
    run_pipeline(args.config)
```

### Integration Test Structure
```python
# tests/test_integration.py
"""End-to-end integration test for the full federated IDS pipeline.

Uses synthetic CSV data mimicking CICIDS2017 column structure to test
the complete pipeline without requiring real dataset download.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.conftest import _CICIDS_COLUMNS


@pytest.fixture
def integration_env(tmp_path):
    """Set up a complete temporary environment for pipeline integration test."""
    # Generate synthetic CSV (~500 rows)
    rng = np.random.RandomState(42)
    n_rows = 500
    data = {}
    for col in _CICIDS_COLUMNS:
        data[col] = rng.rand(n_rows).astype(np.float64) * 1000

    # Inject inf/NaN like real data
    data["Flow Bytes/s"][5] = np.inf
    data[" Flow Packets/s"][10] = np.inf
    data[" Flow Duration"][15] = np.nan

    # Labels: ~60% BENIGN, ~40% DDoS
    labels = ["BENIGN"] * 300 + ["DDoS"] * 200
    rng.shuffle(labels)
    data[" Label"] = labels

    # Write CSV
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    df = pd.DataFrame(data)
    df.to_csv(raw_dir / "synthetic.csv", index=False)

    # Write config
    config = {
        "data": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "files": ["synthetic.csv"],
            "test_size": 0.2,
            "target_features": 30,
            "correlation_threshold": 0.95,
            "variance_threshold": 1e-10,
        },
        "model": {"hidden_layers": [16, 8], "dropout": 0.1, "num_classes": 2},
        "training": {
            "learning_rate": 0.001,
            "local_epochs": 1,
            "batch_size": 32,
            "weighted_loss": True,
        },
        "federation": {"num_clients": 2, "num_rounds": 2, "fraction_fit": 1.0},
        "seed": 42,
        "output_dir": str(tmp_path / "outputs"),
        "log_level": "WARNING",
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path), config


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_pipeline_produces_all_outputs(self, integration_env):
        config_path, config = integration_env
        from federated_ids.pipeline import run_pipeline
        run_pipeline(config_path)

        output_dir = config["output_dir"]
        # Assert checkpoint exists
        assert os.path.isfile(os.path.join(output_dir, "checkpoints", "global_model.pt"))
        # Assert metrics JSON valid
        metrics_path = os.path.join(output_dir, "metrics", "fl_metrics.json")
        assert os.path.isfile(metrics_path)
        with open(metrics_path) as f:
            data = json.load(f)
        assert "rounds" in data
        assert "config" in data
        # Assert plots exist and are non-empty
        plots_dir = os.path.join(output_dir, "plots")
        for png in ["confusion_matrix.png", "convergence.png", "client_comparison.png"]:
            path = os.path.join(plots_dir, png)
            assert os.path.isfile(path), f"Missing: {png}"
            assert os.path.getsize(path) > 0, f"Empty: {png}"
        # Assert classification report
        assert os.path.isfile(os.path.join(plots_dir, "classification_report.txt"))
```

### Console Script Registration
```toml
# In pyproject.toml [project.scripts] -- add one line:
federated-ids-run-all = "federated_ids.pipeline:main"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Shell scripts chaining commands | Python orchestrator calling functions | Standard practice | Better error handling, cross-platform, same-process |
| Manual README maintenance | README as part of codebase with embedded images | Standard practice | Screenshots in docs/ folder, referenced via relative paths |

**Deprecated/outdated:**
- None relevant. This phase uses established Python packaging and testing patterns that are stable.

## Open Questions

1. **eval.__main__:main() logging.basicConfig() interaction**
   - What we know: `main()` calls `logging.basicConfig()` internally. When called after the pipeline runner has already configured logging, this is a no-op (safe).
   - What's unclear: Whether any stage's logging setup might override the root logger format. In practice, `basicConfig()` is a no-op after first call, so this should be fine.
   - Recommendation: Test confirms logging works correctly. No action needed.

2. **Screenshot generation for README**
   - What we know: The user wants actual output screenshots (confusion matrix, convergence plot) committed to docs/ and embedded in README.
   - What's unclear: These can only be generated after running the pipeline with real data. The planner should create a task that assumes the user has run the pipeline once and provides instructions for capturing/committing the screenshots.
   - Recommendation: Create placeholder references in README, add instructions for the user to generate and commit them after first run.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `.venv/Scripts/python -m pytest tests/test_integration.py -x -v` |
| Full suite command | `.venv/Scripts/python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODL-03 | Best global model checkpoint saved during FL training | integration | `.venv/Scripts/python -m pytest tests/test_integration.py::TestFullPipeline::test_pipeline_produces_all_outputs -x` | Wave 0 |
| INFR-02 | Reproducible pipeline with fixed seeds and documented config | integration | `.venv/Scripts/python -m pytest tests/test_integration.py::TestFullPipeline::test_pipeline_produces_all_outputs -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/Scripts/python -m pytest tests/test_integration.py -x -v`
- **Per wave merge:** `.venv/Scripts/python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_integration.py` -- end-to-end pipeline integration test (covers MODL-03, INFR-02)
- [ ] Integration test fixture generating synthetic CICIDS2017 CSV with ~500 rows

*(Existing test infrastructure covers all unit-level concerns. Only the new integration test file is needed.)*

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis of all 21 Python source files in `src/federated_ids/`
- Direct analysis of all 10 test files in `tests/` (88 tests total)
- `pyproject.toml` -- dependency versions, console script registration pattern
- `config/default.yaml` -- full configuration structure
- `tests/conftest.py` -- `_CICIDS_COLUMNS` list (78 columns with whitespace quirks)

### Secondary (MEDIUM confidence)
- Python `logging.basicConfig()` behavior: documented in Python stdlib docs that subsequent calls are no-ops if root logger has handlers

### Tertiary (LOW confidence)
- None. All findings are based on direct codebase analysis.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new libraries, all patterns verified in existing codebase
- Architecture: HIGH - pipeline runner is thin wrapper around existing verified entry points
- Pitfalls: HIGH - identified from direct code reading (argparse conflicts, logging clobber, etc.)
- Integration test design: HIGH - follows established patterns from conftest.py and test_fl.py

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable -- no external dependency changes expected)
