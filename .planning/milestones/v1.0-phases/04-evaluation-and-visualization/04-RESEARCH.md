# Phase 4: Evaluation and Visualization - Research

**Researched:** 2026-03-09
**Domain:** ML evaluation metrics, publication-quality plotting, TensorBoard integration
**Confidence:** HIGH

## Summary

Phase 4 adds post-training evaluation and visualization to the federated IDS. The codebase already has all the building blocks: the `evaluate()` function in `model/train.py` returns predictions/labels, `fl_metrics.json` stores per-round convergence data, and `partition_iid()` can recreate client splits. The work is primarily about composing these existing pieces with well-established visualization libraries (matplotlib 3.10.8, seaborn 0.13.2, scikit-learn 1.8.0) that are already installed, plus adding TensorBoard (not yet installed) for live monitoring.

The evaluation module (`src/federated_ids/eval/`) already has an `__init__.py` placeholder. All plotting uses the non-interactive `Agg` backend (verified working) and saves to `outputs/plots/`. The confusion matrix requires custom annotations combining percentages and counts in each cell. The convergence plots read from the existing `fl_metrics.json`. The per-client comparison requires training local-only models and comparing them against the global model. TensorBoard integration modifies `run_federated_training()` in `fl/server.py` to add `SummaryWriter` calls inside the FL loop.

**Primary recommendation:** Implement as two plans -- (1) all static evaluation and plotting (confusion matrix, classification report, convergence plots, per-client comparison, CLI entry point), and (2) TensorBoard integration into the FL loop, since TensorBoard modifies existing server code while everything else is new code in the eval module.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- seaborn 'whitegrid' theme, muted color palette, clear axis labels, no decorative elements
- Figure size: 10x6 inches for all plots
- DPI: 150 for saved PNG files
- Save all plots to `outputs/plots/`
- Confusion matrix: show both normalized percentages and absolute counts in cells (e.g. "95.2% (1423)")
- Blues colormap for heatmap
- Class labels: "BENIGN" / "DDoS" on axes
- Classification report: print to console AND save to `outputs/plots/classification_report.txt`
- Per-client comparison: grouped bar chart, one group per metric (F1, Precision, Recall), one bar per client
- Per-client: evaluate each client's locally-trained model AND the global model on the held-out test set
- Convergence plots: global metrics only (no per-client overlay)
- Convergence: two subplots side by side: loss over rounds + F1 over rounds
- Convergence data sourced from `fl_metrics.json`
- TensorBoard: scalars only (loss, accuracy, F1, precision, recall per round)
- TensorBoard: integrated into FL loop inside `run_federated_training()` after each round
- TensorBoard log files stored at `outputs/tensorboard/`
- TensorBoard: global metrics only (no per-client TensorBoard logging)
- `tensorboard` package needs to be added to pyproject.toml dependencies

### Claude's Discretion
- Exact seaborn/matplotlib styling details (font sizes, tick formatting, grid density)
- Loading skeleton for confusion matrix annotations
- How to structure the eval module internally (single file vs multiple)
- TensorBoard run naming convention

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-02 | Generate confusion matrix and classification report on held-out test set | sklearn.metrics.confusion_matrix + classification_report (1.8.0), seaborn.heatmap with custom annot array for dual percentage/count display, evaluate global model checkpoint on X_test/y_test |
| EVAL-03 | Save convergence plots (loss and F1 over FL rounds) as PNG | matplotlib subplots from fl_metrics.json, two side-by-side panels (loss + F1 vs rounds), seaborn whitegrid styling |
| EVAL-04 | Log training metrics to TensorBoard for real-time monitoring | torch.utils.tensorboard.SummaryWriter.add_scalar inside run_federated_training() loop, tensorboard package dependency |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.8.0 | confusion_matrix, classification_report | Already installed; standard ML evaluation toolkit |
| matplotlib | 3.10.8 | Plot rendering, subplots, savefig | Already installed; base plotting library |
| seaborn | 0.13.2 | Heatmap, styling, color palettes | Already installed; publication-quality statistical viz |
| tensorboard | (to install) | Live metric logging during FL training | PyTorch-native via torch.utils.tensorboard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.26+ | Array manipulation for annotation matrices | Already installed; used throughout codebase |
| torch | 2.10.0 | Model loading, SummaryWriter integration | Already installed; core framework |
| json (stdlib) | - | Reading fl_metrics.json | Already used in server.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| seaborn heatmap | sklearn.metrics.ConfusionMatrixDisplay | sklearn display lacks dual count+percentage annotation; seaborn heatmap is more flexible |
| tensorboard | MLflow (EVLX-03) | MLflow is v2 scope; tensorboard is lightweight, PyTorch-native |
| manual annotation array | seaborn annot=True | annot=True only shows raw values; custom annotation array needed for "95.2% (1423)" format |

**Installation:**
```bash
pip install tensorboard
```

Also update `pyproject.toml` dependencies to add `"tensorboard>=2.14.0"`.

## Architecture Patterns

### Recommended Project Structure
```
src/federated_ids/eval/
    __init__.py          # Already exists (placeholder)
    evaluate.py          # Confusion matrix, classification report, per-client comparison
    plots.py             # Convergence plots, visual styling utilities
    __main__.py          # CLI entry point: federated-ids-evaluate
```

**Rationale for two-file split:** `evaluate.py` handles model loading, inference, and metric computation (heavyweight, imports torch). `plots.py` handles matplotlib/seaborn rendering from JSON data (lightweight, no torch needed for convergence plots). The `__main__.py` orchestrates both. This keeps responsibilities clear and allows convergence plots to be regenerated without loading PyTorch.

### Pattern 1: Non-Interactive Backend for Headless Rendering
**What:** Set matplotlib backend to 'Agg' before any pyplot import to avoid GUI dependency
**When to use:** Always -- this runs on servers/CI without display
**Example:**
```python
# Source: Verified working in project .venv (matplotlib 3.10.8)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
```

### Pattern 2: Custom Annotation Array for Dual Count/Percentage
**What:** Build a numpy array of formatted strings for seaborn heatmap annot parameter
**When to use:** When cells need to show both percentage and absolute count
**Example:**
```python
# Source: seaborn 0.13.2 heatmap API (annot accepts array-like of strings)
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum() * 100

# Build annotation strings: "95.2%\n(1423)"
labels = np.array([
    [f"{pct:.1f}%\n({count})" for pct, count in zip(pct_row, count_row)]
    for pct_row, count_row in zip(cm_norm, cm)
])

sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
            xticklabels=["BENIGN", "DDoS"],
            yticklabels=["BENIGN", "DDoS"])
```

### Pattern 3: Reuse Existing evaluate() for All Inference
**What:** Load checkpoint, create model, call existing `evaluate()` from `model/train.py`
**When to use:** For confusion matrix evaluation and per-client evaluation
**Example:**
```python
from federated_ids.model.train import evaluate
from federated_ids.model.model import MLP

model = MLP(input_dim, hidden_layers, num_classes, dropout)
model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
model.to(device)
model.eval()

# For confusion matrix, need raw predictions -- extend or call evaluate()
# then also gather y_true/y_pred arrays
```

### Pattern 4: Per-Client Local Training for Comparison
**What:** For each client, train a local-only model and evaluate on the held-out test set
**When to use:** Per-client comparison chart
**Example:**
```python
# Reuse existing FL building blocks
from federated_ids.data.partition import partition_iid, create_dataloaders
from federated_ids.fl.client import FederatedClient
from federated_ids.model.model import MLP

# Train each client locally for same total epochs as FL
# Evaluate each local model on test_loader
# Compare against global model evaluated on same test_loader
```

### Pattern 5: TensorBoard SummaryWriter in FL Loop
**What:** Create SummaryWriter before loop, log scalars per round, close after loop
**When to use:** Inside run_federated_training()
**Example:**
```python
# Source: torch.utils.tensorboard (PyTorch 2.10 docs)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

for round_num in range(1, num_rounds + 1):
    # ... existing FL logic ...
    metrics = server_evaluate(...)

    writer.add_scalar("Global/loss", metrics["loss"], round_num)
    writer.add_scalar("Global/accuracy", metrics["accuracy"], round_num)
    writer.add_scalar("Global/f1", metrics["f1"], round_num)
    writer.add_scalar("Global/precision", metrics["precision"], round_num)
    writer.add_scalar("Global/recall", metrics["recall"], round_num)

writer.close()
```

### Anti-Patterns to Avoid
- **plt.show() in headless code:** Never call `plt.show()` -- only `plt.savefig()`. The Agg backend does not support display.
- **Forgetting plt.close():** After each savefig, call `plt.close(fig)` to free memory. Multiple plots in one session will leak memory otherwise.
- **Not normalizing confusion matrix for percentage:** Must divide by row sums (for recall-style) or total (for overall percentage). User wants total-based percentages.
- **Using annot=True with fmt for dual display:** `annot=True` only shows raw values. Must pass a custom string array and use `fmt=""` (empty string format).
- **SummaryWriter without close():** Causes unflushed data. Always call `writer.close()` in a finally block or use a context manager pattern.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Confusion matrix computation | Manual TP/FP/TN/FN counting | `sklearn.metrics.confusion_matrix(y_true, y_pred)` | Handles edge cases, multi-class, normalization modes |
| Classification report text | Manual metric formatting | `sklearn.metrics.classification_report(y_true, y_pred, target_names=["BENIGN", "DDoS"])` | Consistent formatting, handles zero-division, support/weighted avg |
| Heatmap rendering | Manual matplotlib imshow + text | `seaborn.heatmap(cm, annot=labels, fmt="", cmap="Blues")` | Handles colorbar, text contrast, axis labels, font sizing |
| TensorBoard logging | Custom file-based metric logging | `torch.utils.tensorboard.SummaryWriter` | Standard format, browseable in TensorBoard UI, async writing |
| Prediction collection | Re-implementing inference loop | Extend existing `evaluate()` to optionally return y_true/y_pred | Existing function already has the complete inference loop |

**Key insight:** The existing `evaluate()` function returns summary metrics (dict with loss, accuracy, f1, precision, recall) but not the raw y_true/y_pred arrays. For the confusion matrix, we need those arrays. Rather than duplicating the inference loop, create an `evaluate_detailed()` that calls the same inference logic but also returns the raw predictions and labels. This avoids code duplication and ensures metrics consistency.

## Common Pitfalls

### Pitfall 1: Double Inference for Confusion Matrix
**What goes wrong:** Running model inference twice -- once for metrics, once for confusion matrix
**Why it happens:** The existing `evaluate()` returns aggregated metrics but not raw predictions
**How to avoid:** Create a single `evaluate_detailed()` function that returns both the metric dict AND the raw y_true/y_pred arrays. The confusion matrix, classification report, and summary metrics all come from one inference pass.
**Warning signs:** Two separate model.eval() + torch.no_grad() blocks doing the same thing

### Pitfall 2: Confusion Matrix Normalization Direction
**What goes wrong:** Normalizing by columns instead of by total, or by rows
**Why it happens:** Confusion about what percentage to show
**How to avoid:** User specified "95.2% (1423)" meaning percentage of total dataset. Use `cm / cm.sum() * 100` for overall percentage. If per-class recall is desired, use `normalize='true'` in sklearn. Since user wants absolute counts alongside percentages, compute both from the raw confusion matrix.
**Warning signs:** Percentages in each row summing to 100% (that's recall normalization, not total)

### Pitfall 3: Per-Client Training Epoch Count
**What goes wrong:** Training each local model for only `local_epochs` (1), resulting in undertrained local models that make the federation comparison unfair
**Why it happens:** Confusing FL local epochs with total training effort
**How to avoid:** Each client's local-only model should train for `num_rounds * local_epochs` total epochs to match the total training budget of the federated approach. This makes the comparison fair: same total compute, different coordination strategy.
**Warning signs:** Every local model performs terribly and federation "wins" trivially

### Pitfall 4: TensorBoard Import Guard
**What goes wrong:** `from torch.utils.tensorboard import SummaryWriter` fails if tensorboard package not installed
**Why it happens:** tensorboard is an optional dependency for PyTorch
**How to avoid:** Add tensorboard to pyproject.toml dependencies. Also consider a try/except import with graceful degradation (log a warning and skip TB logging if not available). This prevents the entire FL pipeline from breaking if someone hasn't installed tensorboard.
**Warning signs:** ImportError at runtime, or FL training that worked in Phase 3 suddenly breaks

### Pitfall 5: Matplotlib State Pollution Between Plots
**What goes wrong:** Second plot inherits styling or data from first plot
**Why it happens:** matplotlib uses global state (current figure, current axes)
**How to avoid:** Always use `fig, ax = plt.subplots()` pattern (explicit figure/axes), and call `plt.close(fig)` after `fig.savefig()`. Never use `plt.plot()` directly on the implicit global axes.
**Warning signs:** Unexpected axis labels, wrong color scheme on second plot

### Pitfall 6: Classification Report LaTeX Compatibility
**What goes wrong:** Classification report text has variable spacing that breaks LaTeX table formatting
**Why it happens:** `classification_report()` uses space-padded fixed-width text
**How to avoid:** Save the default text output (human-readable), but also note that `output_dict=True` returns a dict that could be formatted into LaTeX. For the v1 requirement, the text file is sufficient. The user noted it should be "copy-pasteable into a LaTeX table" -- the fixed-width output from sklearn is close but may need manual adjustment. Consider also saving a CSV or the dict form.
**Warning signs:** User complains about formatting in thesis

## Code Examples

### Detailed Evaluation with Raw Predictions
```python
# Extends existing evaluate() pattern from model/train.py
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
)

def evaluate_detailed(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate model and return metrics plus raw predictions.

    Returns:
        Tuple of (metrics_dict, y_true, y_pred).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / len(data_loader.dataset)

    metrics = {
        "loss": avg_loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
    }

    return metrics, y_true, y_pred
```

### Confusion Matrix Plot with Dual Annotation
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Save confusion matrix with percentage and count in each cell."""
    sns.set_theme(style="whitegrid", palette="muted")

    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    cm_pct = cm / total * 100

    labels = np.array([
        [f"{pct:.1f}%\n({count})" for pct, count in zip(pct_row, count_row)]
        for pct_row, count_row in zip(cm_pct, cm)
    ])

    class_names = ["BENIGN", "DDoS"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, square=False, linewidths=0.5,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### Convergence Plot from fl_metrics.json
```python
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_convergence(metrics_path, output_path):
    """Two side-by-side subplots: loss and F1 over FL rounds."""
    sns.set_theme(style="whitegrid", palette="muted")

    with open(metrics_path) as f:
        data = json.load(f)

    rounds_data = data["rounds"]
    round_nums = [r["round"] for r in rounds_data]
    losses = [r["loss"] for r in rounds_data]
    f1_scores = [r["f1"] for r in rounds_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    ax1.plot(round_nums, losses, marker="o", linewidth=2)
    ax1.set_xlabel("FL Round", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss over FL Rounds", fontsize=13)

    ax2.plot(round_nums, f1_scores, marker="o", linewidth=2, color=sns.color_palette("muted")[1])
    ax2.set_xlabel("FL Round", fontsize=12)
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_title("F1 Score over FL Rounds", fontsize=13)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### Per-Client Comparison Grouped Bar Chart
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_client_comparison(client_metrics, global_metrics, output_path):
    """Grouped bar chart: F1, Precision, Recall per client + global model.

    Args:
        client_metrics: List of dicts, one per client, each with f1/precision/recall.
        global_metrics: Dict with f1/precision/recall for the global model.
    """
    sns.set_theme(style="whitegrid", palette="muted")

    metric_names = ["F1", "Precision", "Recall"]
    metric_keys = ["f1", "precision", "recall"]

    n_clients = len(client_metrics)
    labels = [f"Client {i}" for i in range(n_clients)] + ["Global"]
    all_metrics = client_metrics + [global_metrics]

    x = np.arange(len(metric_names))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("muted", n_colors=len(labels))

    for i, (label, metrics) in enumerate(zip(labels, all_metrics)):
        values = [metrics[k] for k in metric_keys]
        offset = (i - len(labels) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=colors[i])

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Client vs Global Model Performance", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### TensorBoard Integration in FL Loop
```python
# Added to run_federated_training() in fl/server.py
import os

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False

# Inside run_federated_training():
tb_dir = os.path.join(output_dir, "tensorboard")
writer = SummaryWriter(log_dir=tb_dir) if _HAS_TENSORBOARD else None

for round_num in range(1, num_rounds + 1):
    # ... existing client training + aggregation ...
    metrics = server_evaluate(...)

    if writer is not None:
        writer.add_scalar("Global/loss", metrics["loss"], round_num)
        writer.add_scalar("Global/accuracy", metrics["accuracy"], round_num)
        writer.add_scalar("Global/f1", metrics["f1"], round_num)
        writer.add_scalar("Global/precision", metrics["precision"], round_num)
        writer.add_scalar("Global/recall", metrics["recall"], round_num)

    # ... existing checkpoint logic ...

# After loop:
if writer is not None:
    writer.close()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| sklearn ConfusionMatrixDisplay | seaborn.heatmap with custom annotations | N/A (user choice) | More flexible dual annotation format |
| tensorboardX (third-party) | torch.utils.tensorboard (built-in) | PyTorch 1.2+ (2019) | No extra dependency beyond tensorboard package |
| plt.plot() global API | fig, ax = plt.subplots() OOP API | Best practice since matplotlib 3.x | Avoids state pollution, better for multiple plots |

**Deprecated/outdated:**
- `tensorboardX`: Replaced by `torch.utils.tensorboard` which is built into PyTorch. Do not use tensorboardX.
- `sklearn.metrics.plot_confusion_matrix()`: Deprecated in sklearn 1.2, removed in 1.4. Use `ConfusionMatrixDisplay.from_predictions()` or (in our case) seaborn heatmap for more control.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | `pyproject.toml` ([tool.pytest.ini_options]) |
| Quick run command | `.venv/Scripts/python.exe -m pytest tests/test_eval.py -x -q` |
| Full suite command | `.venv/Scripts/python.exe -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-02 | evaluate_detailed returns metrics + y_true/y_pred | unit | `pytest tests/test_eval.py::TestEvaluateDetailed -x` | Wave 0 |
| EVAL-02 | Confusion matrix PNG is saved with correct dimensions | unit | `pytest tests/test_eval.py::TestConfusionMatrixPlot -x` | Wave 0 |
| EVAL-02 | Classification report text file is saved | unit | `pytest tests/test_eval.py::TestClassificationReport -x` | Wave 0 |
| EVAL-03 | Convergence plot reads fl_metrics.json and saves PNG | unit | `pytest tests/test_eval.py::TestConvergencePlot -x` | Wave 0 |
| EVAL-03 | Convergence plot has two subplots (loss + F1) | unit | `pytest tests/test_eval.py::TestConvergencePlot -x` | Wave 0 |
| EVAL-04 | TensorBoard SummaryWriter logs 5 scalars per round | unit | `pytest tests/test_eval.py::TestTensorBoardLogging -x` | Wave 0 |
| EVAL-04 | TensorBoard gracefully degrades if package missing | unit | `pytest tests/test_eval.py::TestTensorBoardFallback -x` | Wave 0 |
| ALL | Per-client comparison bar chart saved as PNG | unit | `pytest tests/test_eval.py::TestClientComparison -x` | Wave 0 |
| ALL | CLI entry point runs without error | unit | `pytest tests/test_eval.py::TestEvalCLI -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/Scripts/python.exe -m pytest tests/test_eval.py -x -q`
- **Per wave merge:** `.venv/Scripts/python.exe -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_eval.py` -- covers EVAL-02, EVAL-03, EVAL-04
- [ ] Framework install: `pip install tensorboard` -- not yet in environment

### Testing Strategy Notes
- **Plot tests:** Verify file existence, file size > 0, and optionally image dimensions via PIL. Do NOT pixel-compare plots.
- **Confusion matrix tests:** Use synthetic data with known y_true/y_pred to verify correct metric values and file output.
- **TensorBoard tests:** Use `unittest.mock.patch` or check for event file existence in a tmp directory. Mock SummaryWriter if tensorboard not available in CI.
- **Pattern consistency:** Follow existing test patterns -- class-based tests, fixtures from conftest.py, tmp_path for output directories.

## Open Questions

1. **evaluate_detailed vs extending evaluate()**
   - What we know: The existing `evaluate()` in `model/train.py` computes y_true/y_pred internally but only returns aggregated metrics
   - What's unclear: Whether to modify `evaluate()` to optionally return raw predictions (adds a parameter) or create a separate `evaluate_detailed()` function (code duplication risk)
   - Recommendation: Create `evaluate_detailed()` in `eval/evaluate.py` that mirrors the inference logic but returns the raw arrays. This avoids modifying the existing function signature that FL code depends on, and keeps Phase 3 code stable.

2. **Per-client local training epochs**
   - What we know: FL uses `num_rounds * local_epochs` total local updates. A fair local baseline should match total compute.
   - What's unclear: Exactly how many epochs for local training -- `num_rounds * local_epochs` (default: 20*1=20)?
   - Recommendation: Use `num_rounds * local_epochs` as the local training epoch count. This is the standard fair comparison in FL literature.

3. **TensorBoard run naming**
   - What we know: Runs should be distinguishable if user runs training multiple times
   - What's unclear: Whether to use timestamp, config hash, or simple name
   - Recommendation: Use a subdirectory like `outputs/tensorboard/fl_run` (simple, user decision area). TensorBoard's UI handles multiple runs in the same directory naturally.

## Sources

### Primary (HIGH confidence)
- scikit-learn 1.8.0 installed in project .venv -- confusion_matrix params: `[y_true, y_pred, labels, sample_weight, normalize]`; classification_report params: `[y_true, y_pred, labels, target_names, sample_weight, digits, output_dict, zero_division]`
- matplotlib 3.10.8 installed in project .venv -- Agg backend verified working
- seaborn 0.13.2 installed in project .venv -- heatmap params verified: `[data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, linewidths, ...]`
- PyTorch 2.10.0 installed in project .venv -- torch.utils.tensorboard.SummaryWriter requires tensorboard package
- [scikit-learn confusion_matrix docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [scikit-learn classification_report docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [seaborn heatmap docs](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [PyTorch tensorboard docs](https://docs.pytorch.org/docs/stable/tensorboard.html)

### Secondary (MEDIUM confidence)
- [PyTorch TensorBoard tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) -- verified SummaryWriter usage pattern
- Existing codebase patterns (evaluate(), server.py FL loop, conftest.py fixtures) -- directly inspected

### Tertiary (LOW confidence)
- None -- all findings verified against installed packages or official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed (except tensorboard), versions verified, API signatures confirmed via introspection
- Architecture: HIGH -- builds directly on existing codebase patterns (evaluate, partition_iid, FL loop), clear integration points identified
- Pitfalls: HIGH -- based on direct code inspection and well-known matplotlib/seaborn patterns
- TensorBoard: HIGH -- standard PyTorch integration, API stable since PyTorch 1.2

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable libraries, no breaking changes expected)
