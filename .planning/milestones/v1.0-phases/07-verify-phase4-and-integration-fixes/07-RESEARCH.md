# Phase 7: Verify Phase 4 & Integration Fixes - Research

**Researched:** 2026-03-10
**Domain:** PyTorch evaluation verification, TensorBoard event inspection, config-driven loss selection
**Confidence:** HIGH

## Summary

Phase 7 is a verification and bug-fix phase. It has three distinct work streams: (1) formally verifying EVAL-02, EVAL-03, EVAL-04 by running the existing Phase 4 code paths with synthetic data and checking outputs, (2) fixing the `standalone_train` weighted_loss bug in `model/train.py`, and (3) producing a VERIFICATION.md in the Phase 6 format.

The codebase is mature and well-structured. All evaluation functions (`evaluate_detailed`, plot functions, TensorBoard integration) already exist and work -- this phase just needs to exercise them in a controlled environment and document the results. The weighted_loss bug is a 5-line fix where `standalone_train` unconditionally applies class-weighted CrossEntropyLoss (lines 323-333 of `model/train.py`) instead of checking `training_config.get("weighted_loss", False)` like `server.py` (lines 353-354) and `eval/__main__.py` (lines 128-139) already do.

**Primary recommendation:** Follow the Phase 6 `scripts/verify_phase1.py` pattern exactly -- a standalone script that generates synthetic data, exercises each requirement's code path, captures pass/fail with measured values, and cleans up after itself. Use `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` (already a project dependency via `tensorboard>=2.14.0`) to read tfevents files for EVAL-04 verification -- no new dependencies needed.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- TensorBoard verification (EVAL-04): Run mini FL training (2 clients, 2 rounds, tiny synthetic data) with TB enabled; verify tfevents file exists in runs/ directory AND contains expected scalar tags (accuracy, f1, loss); tensorboard is a required project dependency -- if not installed, check FAILS (not skipped); clean up temporary TB event files after verification
- Plot & report verification (EVAL-02, EVAL-03): EVAL-02 runs eval module with synthetic model + synthetic test data, verify confusion matrix PNG and classification report text produced with expected structure; EVAL-03 checks convergence PNGs exist, have non-zero file size, and can be opened by PIL/matplotlib as valid images; per-client comparison visualization also verified; all verification outputs in temp directory, checked, then cleaned up; evidence captured in VERIFICATION.md
- weighted_loss fix: Fix standalone_train only; when weighted_loss is false (or absent), use plain CrossEntropyLoss() without weights; add pytest regression test; update config/default.yaml to explicitly include weighted_loss: true
- Verification format: Follow Phase 6 pattern with standalone script at scripts/verify_phase4.py; pass/fail with actual measured values; VERIFICATION.md with traceability table; script is re-runnable

### Claude's Discretion
- Internal structure of the verification script
- How synthetic data and model are generated for eval verification
- Exact format of human-readable output
- How to parse/check TB event file scalar tags

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-02 | Generate confusion matrix and classification report on held-out test set | `eval/plots.py` has `plot_confusion_matrix()` and `save_classification_report()` -- verification script calls them with synthetic y_true/y_pred, checks PNG exists + non-zero size, checks report text contains expected class names |
| EVAL-03 | Save convergence plots (loss and accuracy over FL rounds) as PNG | `eval/plots.py` has `plot_convergence()` reading from fl_metrics.json -- verification script creates synthetic metrics JSON, calls plot_convergence(), validates PNG via PIL.Image.open() |
| EVAL-04 | Log training metrics to TensorBoard for real-time monitoring | `fl/server.py` lines 389-440 integrate SummaryWriter -- verification script runs mini FL training with real TB (not mocked), then reads tfevents via EventAccumulator to verify scalar tags exist |
</phase_requirements>

## Standard Stack

### Core (already in project -- no new dependencies)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| tensorboard | >=2.14.0 | TB event writing + EventAccumulator for reading | Already a project dependency in pyproject.toml |
| torch | >=2.4.0 | Model creation and training for verification | Core project dependency |
| matplotlib | >=3.9.0 | Plot generation (Agg backend) | Core project dependency |
| scikit-learn | >=1.5.0 | confusion_matrix, classification_report | Core project dependency |
| PIL/Pillow | (bundled with matplotlib) | Image validation for PNG verification | Available via matplotlib dependency |
| pytest | >=8.0.0 | Regression test for weighted_loss fix | Dev dependency |

### No New Dependencies Needed
The verification script uses `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` to read tfevents files. This is part of the `tensorboard` package already installed as a required dependency. No external parsing library (e.g., tbparse) is needed.

## Architecture Patterns

### Verification Script Structure (following Phase 6 pattern)
```
scripts/
    verify_phase4.py          # Standalone verification script (new)
src/federated_ids/
    model/train.py            # weighted_loss bug fix (modify lines 323-333)
tests/
    test_train.py             # Add regression test class (append)
config/
    default.yaml              # Already has weighted_loss: true (no change needed)
.planning/phases/07-.../
    07-VERIFICATION.md         # Verification report (new)
```

### Pattern 1: Verification Script Structure (from verify_phase1.py)
**What:** Standalone Python script with individual check functions per requirement, synthetic data generation, summary table, and exit code 0/1.
**When to use:** All verification phases.
**Key elements from verify_phase1.py to replicate:**
```python
#!/usr/bin/env python
"""Phase 4 requirement verification script."""
import os, sys, tempfile, shutil, platform
from datetime import datetime

def check_eval_02(tmp_dir: str) -> dict:
    """EVAL-02: Verify confusion matrix and classification report."""
    # Generate synthetic y_true/y_pred
    # Call plot_confusion_matrix() and save_classification_report()
    # Check file exists, non-zero size, report has expected structure
    return {"req_id": "EVAL-02", "check": "...", "status": "PASS/FAIL", "value": "...", "method": "..."}

def check_eval_03(tmp_dir: str) -> dict:
    """EVAL-03: Verify convergence plots saved as PNG."""
    # Create synthetic fl_metrics.json
    # Call plot_convergence()
    # Verify PNG exists, non-zero, openable by PIL
    return {...}

def check_eval_04(tmp_dir: str) -> dict:
    """EVAL-04: Verify TensorBoard logging."""
    # Run mini FL training with real SummaryWriter
    # Read tfevents via EventAccumulator
    # Check scalar tags exist
    return {...}

def main() -> int:
    # Environment header, run checks, print summary table
    ...
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
```

### Pattern 2: TensorBoard Event File Reading via EventAccumulator
**What:** Use the built-in `EventAccumulator` from tensorboard package to read and validate tfevents files.
**When to use:** EVAL-04 verification.
```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(tb_log_dir)
ea.Reload()
scalar_tags = ea.Tags()["scalars"]
# Expected: ["Global/loss", "Global/accuracy", "Global/f1", "Global/precision", "Global/recall"]
```

### Pattern 3: weighted_loss Conditional Pattern (from server.py lines 352-364)
**What:** Check `training_config.get("weighted_loss", False)` before applying class weights.
**When to use:** The fix for standalone_train.
```python
# CORRECT pattern (server.py, evaluate __main__.py):
weighted_loss = trn_config.get("weighted_loss", False)
if weighted_loss:
    weights_path = os.path.join(processed_dir, "class_weights.json")
    with open(weights_path) as f:
        raw_weights = json.load(f)
    weight_tensor = torch.tensor(
        [raw_weights[str(i)] for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
else:
    criterion = torch.nn.CrossEntropyLoss()
```

### Pattern 4: PNG Validation via PIL
**What:** Open a PNG file with PIL to verify it is a valid image (not corrupt/empty).
**When to use:** EVAL-02 and EVAL-03 verification.
```python
from PIL import Image

img = Image.open(png_path)
width, height = img.size
assert width > 0 and height > 0
# Optionally verify format
assert img.format == "PNG"
```

### Anti-Patterns to Avoid
- **Mocking TensorBoard for EVAL-04 verification:** The unit tests already mock TB. The verification script must use real SummaryWriter + real EventAccumulator to prove the actual integration works end-to-end.
- **Leaving temp files behind:** All verification artifacts must be generated in `tempfile.mkdtemp()` and cleaned up in a `finally` block.
- **Importing from conftest.py in verification script:** The script must be self-contained with inline synthetic data generation (established pattern from Phase 6).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reading TB event files | Custom binary parser | `EventAccumulator` from tensorboard package | tfevents is a protobuf binary format; EventAccumulator is the official reader |
| PNG validation | Custom byte-header checks | `PIL.Image.open()` | Handles all PNG variants, detects truncation/corruption |
| Confusion matrix generation | Manual numpy computation | `sklearn.metrics.confusion_matrix` + existing `plot_confusion_matrix()` | Already implemented in eval/plots.py |
| Classification report | Manual precision/recall calc | `sklearn.metrics.classification_report` + existing `save_classification_report()` | Already implemented in eval/plots.py |

**Key insight:** All evaluation code already exists and works. The verification script just needs to exercise existing functions with synthetic inputs and validate their outputs.

## Common Pitfalls

### Pitfall 1: EventAccumulator Size Guidance
**What goes wrong:** Default `EventAccumulator` only keeps 10000 scalars per tag. With many rounds, some events could be dropped.
**Why it happens:** Size guidance defaults truncate event history.
**How to avoid:** Pass `size_guidance={event_accumulator.SCALARS: 0}` to keep all scalar events. For our 2-round mini training, default is fine, but explicit `0` is safer.
**Warning signs:** `len(ea.Scalars(tag))` returns fewer events than expected rounds.

### Pitfall 2: matplotlib Backend Not Set Before Import
**What goes wrong:** `plt.show()` or display-dependent code fails on headless systems.
**Why it happens:** Default matplotlib backend tries to open a display window.
**How to avoid:** `matplotlib.use('Agg')` must be called before `import matplotlib.pyplot as plt`. The existing `eval/plots.py` already does this correctly at module level. The verification script should also set it before any pyplot usage.
**Warning signs:** `RuntimeError: Invalid DISPLAY variable` or similar.

### Pitfall 3: TB Log Directory Contains Multiple Event Files
**What goes wrong:** EventAccumulator reads only one event file; if multiple exist from previous runs, results may be stale.
**Why it happens:** SummaryWriter creates new event files on each run but doesn't clean old ones.
**How to avoid:** Use a fresh `tempfile.mkdtemp()` for each verification run's TB output directory.
**Warning signs:** Scalar tags or step counts don't match expectations.

### Pitfall 4: standalone_train Reads class_weights.json Unconditionally
**What goes wrong:** When `weighted_loss` is false, the current code still tries to open `class_weights.json`. If the file doesn't exist, it crashes.
**Why it happens:** Lines 323-333 of `model/train.py` don't check the `weighted_loss` flag.
**How to avoid:** The fix must wrap the class_weights.json reading in the same conditional as server.py.
**Warning signs:** `FileNotFoundError` when running standalone_train without class_weights.json.

### Pitfall 5: Regression Test Must Not Depend on Data Pipeline
**What goes wrong:** Test takes too long or fails due to missing CICIDS2017 data.
**Why it happens:** Testing standalone_train end-to-end requires cached tensors from the data pipeline.
**How to avoid:** The regression test should test the criterion construction logic directly (unit test), not invoke the full standalone_train function. Or create synthetic cached tensors in tmp_path like the FL tests do.
**Warning signs:** Test requires real data or takes >30 seconds.

## Code Examples

### The Bug: standalone_train Always Uses Weighted Loss
```python
# model/train.py lines 323-333 (CURRENT -- BUG)
# --- Class-weighted loss ---
weights_path = os.path.join(processed_dir, "class_weights.json")
with open(weights_path) as f:
    raw_weights = json.load(f)

num_classes = model_config["num_classes"]
weight_tensor = torch.tensor(
    [raw_weights[str(i)] for i in range(num_classes)],
    dtype=torch.float32,
).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
```

### The Fix: Match server.py Pattern
```python
# model/train.py lines 323-333 (FIXED -- matches server.py lines 352-364)
# --- Loss function (conditional class weighting) ---
weighted_loss = training_config.get("weighted_loss", False)
if weighted_loss:
    weights_path = os.path.join(processed_dir, "class_weights.json")
    with open(weights_path) as f:
        raw_weights = json.load(f)

    num_classes = model_config["num_classes"]
    weight_tensor = torch.tensor(
        [raw_weights[str(i)] for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    logger.info("Using class-weighted loss: %s (device=%s)", raw_weights, device)
else:
    criterion = torch.nn.CrossEntropyLoss()
    logger.info("Using unweighted loss (device=%s)", device)
```

### EventAccumulator Usage for EVAL-04 Verification
```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def check_eval_04(tmp_dir: str) -> dict:
    """EVAL-04: Verify TensorBoard scalar logging during FL training."""
    # ... run mini FL training with TB enabled (real writer, not mocked) ...

    ea = EventAccumulator(tb_log_dir, size_guidance={"scalars": 0})
    ea.Reload()

    scalar_tags = set(ea.Tags().get("scalars", []))
    expected_tags = {"Global/loss", "Global/accuracy", "Global/f1", "Global/precision", "Global/recall"}

    tags_present = expected_tags.issubset(scalar_tags)

    # Verify each tag has events for each round
    events_per_tag = {}
    for tag in expected_tags:
        if tag in scalar_tags:
            events_per_tag[tag] = len(ea.Scalars(tag))

    all_rounds_logged = all(v == num_rounds for v in events_per_tag.values())

    status = "PASS" if (tags_present and all_rounds_logged) else "FAIL"
    return {
        "req_id": "EVAL-04",
        "check": "TensorBoard scalar logging captures training metrics",
        "status": status,
        "value": f"tags={len(scalar_tags & expected_tags)}/5, events_per_tag={events_per_tag}",
        "method": "run_federated_training() + EventAccumulator tag/event check",
    }
```

### Synthetic Data for Verification (Minimal Pattern)
```python
def _make_synthetic_data(tmp_dir: str, n_features: int = 10) -> tuple:
    """Generate synthetic cached tensors for verification.

    Returns (processed_dir, config) ready for FL training or evaluation.
    """
    rng = np.random.RandomState(42)
    n_train, n_test = 200, 50

    X_train = rng.randn(n_train, n_features).astype(np.float32)
    y_train = (rng.rand(n_train) < 0.3).astype(np.int64)
    X_test = rng.randn(n_test, n_features).astype(np.float32)
    y_test = (rng.rand(n_test) < 0.3).astype(np.int64)

    processed_dir = os.path.join(tmp_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    torch.save(torch.tensor(X_train), os.path.join(processed_dir, "X_train.pt"))
    torch.save(torch.tensor(y_train), os.path.join(processed_dir, "y_train.pt"))
    torch.save(torch.tensor(X_test), os.path.join(processed_dir, "X_test.pt"))
    torch.save(torch.tensor(y_test), os.path.join(processed_dir, "y_test.pt"))

    return processed_dir
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `from tensorboard import SummaryWriter` | `from torch.utils.tensorboard import SummaryWriter` | PyTorch 1.2+ | Project already uses correct import in server.py |
| Read TB events with `tf.compat.v1.train.summary_iterator` | `EventAccumulator` from tensorboard package | tensorboard 2.x | No TensorFlow dependency needed; tensorboard standalone works |

**No deprecated/outdated patterns detected** in the existing codebase relevant to this phase.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-02 | Confusion matrix and classification report generated | smoke (verification script) | `python scripts/verify_phase4.py` | No -- Wave 0 |
| EVAL-03 | Convergence plots saved as valid PNG | smoke (verification script) | `python scripts/verify_phase4.py` | No -- Wave 0 |
| EVAL-04 | TensorBoard logs contain expected scalar tags | smoke (verification script) | `python scripts/verify_phase4.py` | No -- Wave 0 |
| BUG-FIX | standalone_train respects weighted_loss flag | unit (regression test) | `python -m pytest tests/test_train.py::TestWeightedLossConfig -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/ -x -q` (< 10s)
- **Per wave merge:** `python -m pytest tests/ -q` + `python scripts/verify_phase4.py`
- **Phase gate:** Full suite green + verification script exit code 0

### Wave 0 Gaps
- [ ] `scripts/verify_phase4.py` -- covers EVAL-02, EVAL-03, EVAL-04 verification
- [ ] `tests/test_train.py::TestWeightedLossConfig` -- regression test for weighted_loss fix (append to existing file)
- [ ] `.planning/phases/07-.../07-VERIFICATION.md` -- verification report

## Open Questions

1. **PIL availability as standalone import**
   - What we know: PIL/Pillow is bundled as a dependency of matplotlib, so `from PIL import Image` should work. The project does not list Pillow explicitly in pyproject.toml.
   - What's unclear: Whether the matplotlib installation always installs Pillow as a proper importable package.
   - Recommendation: Use `matplotlib.image.imread()` as a fallback if PIL import fails. Both validate that the PNG is a valid image file. In practice, Pillow is always available when matplotlib is installed.

2. **EventAccumulator log_dir vs event file path**
   - What we know: `EventAccumulator` can accept either a directory path or a specific event file path. When given a directory, it reads all event files in that directory.
   - What's unclear: If the TB log directory contains subdirectories (SummaryWriter sometimes creates date-stamped subdirs), EventAccumulator may not recurse.
   - Recommendation: Pass the exact directory that SummaryWriter was configured with (`output_dir/tensorboard`). If SummaryWriter creates a subdir, glob for `**/tfevents*` files and pass the parent directory to EventAccumulator.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `src/federated_ids/model/train.py` lines 323-333 (the bug), `src/federated_ids/fl/server.py` lines 352-364 (correct pattern), `src/federated_ids/eval/__main__.py` lines 128-139 (correct pattern)
- Codebase inspection: `src/federated_ids/eval/plots.py` (all 4 plot/report functions)
- Codebase inspection: `scripts/verify_phase1.py` (Phase 6 verification pattern)
- Codebase inspection: `tests/test_fl.py` TestTensorBoardLogging class (mock-based TB tests)
- Codebase inspection: `config/default.yaml` (already has `weighted_loss: true`)
- `pyproject.toml`: `tensorboard>=2.14.0` is a required dependency

### Secondary (MEDIUM confidence)
- [TensorBoard EventAccumulator source](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py) -- API for reading tfevents files
- [tbparse docs on parsing without tbparse](https://tbparse.readthedocs.io/en/latest/pages/raw.html) -- EventAccumulator usage examples

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, no new dependencies
- Architecture: HIGH -- following established Phase 6 verification pattern exactly
- Pitfalls: HIGH -- bug is clearly identified with exact line numbers, fix pattern is established in two other files
- Code examples: HIGH -- derived directly from existing codebase inspection

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable -- no external dependencies changing)
