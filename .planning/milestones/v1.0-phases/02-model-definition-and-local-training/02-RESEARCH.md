# Phase 2: Model Definition and Local Training - Research

**Researched:** 2026-03-09
**Domain:** PyTorch MLP model definition, local training loop, class-weighted loss, metrics, checkpointing
**Confidence:** HIGH

## Summary

Phase 2 builds an MLP model and standalone training loop on top of Phase 1's data pipeline. The existing codebase provides robust infrastructure: config loading, seeding, device detection, data partitioning with DataLoaders, and cached tensor persistence. The model module (`src/federated_ids/model/`) has a placeholder `__init__.py` ready for population.

The technical challenge is moderate. CICIDS2017 Friday DDoS binary classification is well-studied, with MLP models routinely achieving F1 > 0.96 in published literature. The success criteria of F1 > 0.80 in 5 epochs and attack-class recall > 0.70 are comfortably achievable with the configured architecture (128-64-32 hidden layers, dropout 0.3, Adam LR 0.001) and class-weighted cross-entropy loss. The primary complexity lies in correct integration: loading Phase 1 artifacts (class_weights.json, cached tensors), implementing a validation split within a single client's partition, computing per-epoch metrics on validation data (not training batches), and designing the training loop so that `train_one_epoch()` is reusable by Flower clients in Phase 3.

**Primary recommendation:** Build model.py (nn.Module), train.py (training loop with validation), and a `__main__.py` entry point. Use `sklearn.metrics` for F1/precision/recall computation on validation data. Use `sklearn.model_selection.train_test_split` with `stratify` for the client-local train/val split (PyTorch's `random_split` does not support stratified splitting). Design `train_one_epoch()` as a standalone function that Phase 3's Flower client can call directly.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Full metric suite per epoch: loss, accuracy, precision, recall, F1
- Use Python logging module (consistent with Phase 1 patterns)
- Format: `Epoch 3/5 -- loss: 0.142, acc: 0.95, F1: 0.83, prec: 0.87, rec: 0.79`
- Print a summary table after training completes (all epochs at a glance) -- easy to copy into thesis
- Per-epoch metrics computed on validation data (separate evaluation pass), not training batch metrics
- Save model weights only (state_dict) -- no optimizer state, no metadata embedded in .pt file
- Save only when validation F1 improves -- single `best_model.pt` file, overwritten on improvement
- Save to `outputs/checkpoints/best_model.pt` (under configured output_dir)
- Log checkpoint events to console: `Saved best model (epoch 3, F1=0.84)` -- metadata in logs, not in file
- Entry point: both `python -m federated_ids.model.train` and console script (pyproject.toml)
- Add `training.standalone_epochs` config key (default 5), separate from `training.local_epochs` (1) for FL
- Auto-run data pipeline if processed data doesn't exist -- single command from raw data to trained model
- CLI argument `--client-id` (default 0) to select which client partition to train on
- Client-local validation split: split the selected client's partition into train/val
- Add `training.val_split` config key (default 0.2) -- configurable train/val ratio within client data
- Best-model checkpoint based on client-local val F1 (methodologically sound -- no test set leakage)
- After training completes, run one final evaluation on the global test set and log results separately -- shows true generalization

### Claude's Discretion
- Optimizer choice (Adam expected based on config LR, but Claude can decide)
- Exact summary table formatting
- Internal training loop structure (batch iteration, gradient clipping, etc.)
- Test framework and test structure for model/training code
- How to load class weights from Phase 1 artifacts into the loss function

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODL-01 | Define MLP model in PyTorch (3-layer feed-forward, ReLU, dropout, binary classification) | Architecture pattern with configurable hidden_layers from config; nn.Module subclass with Sequential or explicit layers |
| MODL-02 | Implement local PyTorch training loop with configurable hyperparameters | train_one_epoch() + evaluate() functions; Adam optimizer; class-weighted CrossEntropyLoss; per-epoch validation metrics |
| MODL-03 | Implement model checkpointing to save the best-performing model based on F1-score | torch.save(model.state_dict(), path) when val F1 improves; single best_model.pt overwritten |
| DATA-04 | Handle class imbalance via weighted cross-entropy loss for DDoS minority class | Load class_weights.json from Phase 1 processed_dir; convert to torch.FloatTensor; pass as weight= to nn.CrossEntropyLoss; must be on same device as model |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch (nn.Module) | >=2.4.0 | MLP model definition | Already pinned in pyproject.toml; nn.Module is the standard PyTorch model API |
| torch (optim.Adam) | >=2.4.0 | Optimizer | Standard for tabular MLP; matches config LR=0.001 |
| torch (nn.CrossEntropyLoss) | >=2.4.0 | Weighted loss function | Built-in weight parameter for class imbalance; expects raw logits (no softmax needed) |
| sklearn.metrics | >=1.5.0 | F1, precision, recall, accuracy | Already pinned; standard for classification metrics; supports binary averaging |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sklearn.model_selection.train_test_split | >=1.5.0 | Stratified train/val split within client partition | Splitting client data into train/val with preserved class ratios |
| json (stdlib) | - | Load class_weights.json | Reading Phase 1 artifact |
| logging (stdlib) | - | All console output | Consistent with Phase 1 pattern |
| argparse (stdlib) | - | CLI --client-id argument | Consistent with Phase 1 __main__.py pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sklearn.metrics | torchmetrics | torchmetrics adds a dependency; sklearn already available and sufficient for epoch-level metrics |
| Adam | AdamW | AdamW has decoupled weight decay; unnecessary for this MLP with dropout regularization |
| CrossEntropyLoss | BCEWithLogitsLoss | BCE requires single output node; CrossEntropyLoss with 2 classes matches num_classes=2 config and is more standard for classification |

## Architecture Patterns

### Recommended Project Structure
```
src/federated_ids/model/
    __init__.py          # Export MLP class and key functions
    model.py             # MLP nn.Module definition
    train.py             # Training loop, evaluation, standalone entry point
    __main__.py           # Module entry point (python -m federated_ids.model.train)
```

### Pattern 1: Configurable MLP with nn.Module
**What:** MLP class that accepts hidden_layers list, dropout, and input_dim as constructor args. Builds layers dynamically from the config list [128, 64, 32].
**When to use:** Always -- this is the only model architecture for v1.
**Example:**
```python
# Source: PyTorch nn.Module standard pattern
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int],
                 num_classes: int, dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # Raw logits -- no softmax
```

**Critical:** The forward() method returns raw logits, NOT softmax outputs. CrossEntropyLoss applies log-softmax internally. Applying softmax before CrossEntropyLoss is a common bug that degrades training.

### Pattern 2: Separated train_one_epoch() and evaluate() Functions
**What:** Standalone functions (not methods) that take model, dataloader, criterion, optimizer, and device as arguments.
**When to use:** Always -- this separation allows Phase 3 Flower client to call train_one_epoch() directly.
**Example:**
```python
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train model for one epoch. Returns average training loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(train_loader.dataset)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model. Returns dict with loss, accuracy, precision, recall, F1."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    # Compute metrics using sklearn
    metrics = {
        "loss": total_loss / len(data_loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="binary"),
        "precision": precision_score(all_labels, all_preds, average="binary"),
        "recall": recall_score(all_labels, all_preds, average="binary"),
    }
    return metrics
```

### Pattern 3: Class Weight Loading and Device Placement
**What:** Load class_weights.json from Phase 1, convert to tensor, place on correct device BEFORE creating loss function.
**When to use:** When constructing CrossEntropyLoss with weight parameter.
**Example:**
```python
# Load class weights from Phase 1 artifact
weights_path = os.path.join(processed_dir, "class_weights.json")
with open(weights_path) as f:
    class_weights_dict = json.load(f)

# Convert to tensor -- keys are "0" and "1" (string from JSON)
# Order matters: index 0 = class 0 weight, index 1 = class 1 weight
num_classes = config["model"]["num_classes"]
weight_tensor = torch.tensor(
    [class_weights_dict[str(i)] for i in range(num_classes)],
    dtype=torch.float32,
).to(device)  # CRITICAL: must be on same device as model

criterion = nn.CrossEntropyLoss(weight=weight_tensor)
```

**GOTCHA:** The weight tensor MUST be on the same device as the model inputs. If the model is on CUDA and the weight tensor is on CPU, PyTorch raises a RuntimeError. Always call `.to(device)` on the weight tensor before passing it to CrossEntropyLoss.

### Pattern 4: Stratified Client-Local Train/Val Split
**What:** Split a single client's numpy arrays into train/val preserving class ratios.
**When to use:** Before creating DataLoaders for standalone training.
**Example:**
```python
from sklearn.model_selection import train_test_split

# X_client, y_client are numpy arrays from partition_iid()
X_train, X_val, y_train, y_val = train_test_split(
    X_client, y_client,
    test_size=val_split,  # 0.2 from config
    stratify=y_client,
    random_state=seed,
)
# Then wrap in TensorDataset + DataLoader
```

**Why not torch.utils.data.random_split:** PyTorch's random_split does NOT support stratified splitting. For imbalanced data (BENIGN vs DDoS), a non-stratified split could create a validation set with very few attack samples, making F1 unreliable. sklearn's train_test_split with stratify= is the standard solution.

### Anti-Patterns to Avoid
- **Applying softmax before CrossEntropyLoss:** CrossEntropyLoss combines log_softmax + NLLLoss internally. Double-softmax collapses gradients and prevents learning.
- **Computing metrics on training batches:** Training batch metrics are inflated (model just updated on this data). Always run a separate evaluation pass on held-out validation data.
- **Saving entire model with torch.save(model, path):** This pickles the class structure and breaks when code changes. Always save state_dict only.
- **Calling model(x) during evaluation without torch.no_grad():** Wastes memory tracking gradients. Always wrap evaluation in `with torch.no_grad():` and call `model.eval()`.
- **Forgetting model.train() after model.eval():** Dropout and BatchNorm behave differently in train vs eval mode. Must toggle back.
- **Hardcoding input_dim:** The number of features depends on Phase 1's feature selection (which may vary with config). Read it from the data tensor shape: `input_dim = X_train.shape[1]`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| F1/precision/recall | Manual TP/FP/FN counting | sklearn.metrics.f1_score, precision_score, recall_score | Edge cases with zero division, different averaging modes |
| Class weight computation | Manual 1/frequency calculation | Already computed by Phase 1 (sklearn compute_class_weight) | Just load class_weights.json |
| Stratified splitting | Manual index shuffling | sklearn.model_selection.train_test_split(stratify=) | Proven stratification, seed support |
| Learning rate scheduling | Manual LR decay | torch.optim.lr_scheduler (if needed later) | Not needed for 5-epoch standalone training, but available |

**Key insight:** Phase 1 already computed and saved class_weights.json. Phase 2 must LOAD it, not recompute it. This maintains consistency between the weights used in standalone training and later in FL training.

## Common Pitfalls

### Pitfall 1: CrossEntropyLoss Weight Tensor Device Mismatch
**What goes wrong:** RuntimeError: Expected all tensors to be on the same device
**Why it happens:** weight tensor created on CPU, model moved to CUDA/MPS
**How to avoid:** Call `.to(device)` on the weight tensor before passing to CrossEntropyLoss constructor
**Warning signs:** Error only appears on GPU machines, works fine on CPU-only development

### Pitfall 2: Raw Logits vs Softmax Confusion
**What goes wrong:** Model trains but loss barely decreases, accuracy stuck near random
**Why it happens:** Applying nn.Softmax before passing to CrossEntropyLoss, which internally applies log_softmax -- double application collapses gradients
**How to avoid:** Model forward() returns raw logits. CrossEntropyLoss handles the rest. For predictions, use `logits.argmax(dim=1)`.
**Warning signs:** Loss starts reasonable but doesn't decrease. Accuracy stays at class prior (~50-60% for this dataset).

### Pitfall 3: Evaluating Metrics on Training Data
**What goes wrong:** Reported F1 is 0.99 during training but 0.75 on test set -- overfitting not detected
**Why it happens:** Computing metrics on the same data the model was just trained on
**How to avoid:** User decision already locks this: "Per-epoch metrics computed on validation data (separate evaluation pass)". Implement evaluate() on val_loader, not train_loader.
**Warning signs:** Suspiciously high metrics that don't translate to test performance.

### Pitfall 4: Forgetting model.eval() / model.train() Toggle
**What goes wrong:** Dropout still active during validation (noisy metrics) or dropout disabled during training (no regularization)
**Why it happens:** Not toggling eval/train mode
**How to avoid:** evaluate() calls model.eval() at start. train_one_epoch() calls model.train() at start.
**Warning signs:** Validation metrics vary wildly between runs with same seed.

### Pitfall 5: input_dim Hardcoding
**What goes wrong:** Model instantiation crashes or silently trains on wrong feature count
**Why it happens:** Hardcoding input_dim=30 when feature selection produced 25 (or vice versa)
**How to avoid:** Read input_dim from the loaded data tensor: `input_dim = X_train.shape[1]` or from the loaded features.json length
**Warning signs:** Dimension mismatch errors at first forward pass.

### Pitfall 6: Class Weight JSON Key Types
**What goes wrong:** KeyError when loading class_weights.json
**Why it happens:** JSON serializes integer keys as strings ("0", "1"), but code tries to access with int keys
**How to avoid:** Access as `class_weights_dict[str(i)]` or `class_weights_dict["0"]`
**Warning signs:** KeyError: 0 (integer key not found in JSON dict with string keys)

### Pitfall 7: Data Pipeline Not Auto-Running
**What goes wrong:** User runs `python -m federated_ids.model.train` on fresh checkout, gets FileNotFoundError for X_train.pt
**Why it happens:** Cached tensors don't exist yet
**How to avoid:** User decision requires auto-running data pipeline if processed data doesn't exist. Check for cached tensors and call Phase 1's main() if missing.
**Warning signs:** Only works after manual `federated-ids-preprocess` run.

## Code Examples

### Complete Standalone Training Flow
```python
# Source: Synthesized from PyTorch docs + project patterns
def standalone_train(config_path: str = "config/default.yaml", client_id: int = 0):
    """Single-command standalone training: raw data -> trained model."""
    config = load_config(config_path)
    set_global_seed(config["seed"])
    device = get_device()

    # 1. Ensure data exists (auto-run pipeline if needed)
    processed_dir = config["data"]["processed_dir"]
    if not _cache_exists(processed_dir):
        logger.info("Processed data not found. Running data pipeline...")
        from federated_ids.data.preprocess import main as run_pipeline
        run_pipeline(config_path)

    # 2. Load cached tensors
    X_train = torch.load(os.path.join(processed_dir, "X_train.pt"), weights_only=True).numpy()
    y_train = torch.load(os.path.join(processed_dir, "y_train.pt"), weights_only=True).numpy()
    X_test = torch.load(os.path.join(processed_dir, "X_test.pt"), weights_only=True).numpy()
    y_test = torch.load(os.path.join(processed_dir, "y_test.pt"), weights_only=True).numpy()

    # 3. Partition and select client
    partitions = partition_iid(X_train, y_train,
                               num_clients=config["federation"]["num_clients"],
                               seed=config["seed"])
    X_client, y_client = partitions[client_id]

    # 4. Client-local stratified train/val split
    val_split = config["training"].get("val_split", 0.2)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_client, y_client, test_size=val_split,
        stratify=y_client, random_state=config["seed"]
    )

    # 5. Create DataLoaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.int64)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.int64)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.int64)),
        batch_size=batch_size, shuffle=False
    )

    # 6. Model, optimizer, weighted loss
    input_dim = X_tr.shape[1]
    model = MLP(
        input_dim=input_dim,
        hidden_layers=config["model"]["hidden_layers"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["training"]["learning_rate"])

    # Load class weights from Phase 1 artifact
    with open(os.path.join(processed_dir, "class_weights.json")) as f:
        cw = json.load(f)
    weight_tensor = torch.tensor(
        [cw[str(i)] for i in range(config["model"]["num_classes"])],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # 7. Training loop
    epochs = config["training"].get("standalone_epochs", 5)
    best_f1 = 0.0
    epoch_history = []
    checkpoint_dir = os.path.join(config.get("output_dir", "./outputs"), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %d/%d -- loss: %.3f, acc: %.2f, F1: %.2f, prec: %.2f, rec: %.2f",
            epoch, epochs,
            val_metrics["loss"], val_metrics["accuracy"],
            val_metrics["f1"], val_metrics["precision"], val_metrics["recall"],
        )
        epoch_history.append(val_metrics)

        # Checkpoint on F1 improvement
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            logger.info("Saved best model (epoch %d, F1=%.2f)", epoch, best_f1)

    # 8. Summary table
    _print_summary_table(epoch_history)

    # 9. Final global test evaluation
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info("--- Global Test Set Evaluation ---")
    logger.info("  loss: %.3f, acc: %.2f, F1: %.2f, prec: %.2f, rec: %.2f",
                test_metrics["loss"], test_metrics["accuracy"],
                test_metrics["f1"], test_metrics["precision"], test_metrics["recall"])
```

### Summary Table Format
```python
def _print_summary_table(history: list[dict]) -> None:
    """Print a thesis-appendix-ready summary table of all epochs."""
    header = f"{'Epoch':>5} | {'Loss':>7} | {'Acc':>6} | {'F1':>6} | {'Prec':>6} | {'Rec':>6}"
    separator = "-" * len(header)
    logger.info("--- Training Summary ---")
    logger.info(header)
    logger.info(separator)
    for i, m in enumerate(history, 1):
        logger.info(
            "%5d | %7.4f | %6.4f | %6.4f | %6.4f | %6.4f",
            i, m["loss"], m["accuracy"], m["f1"], m["precision"], m["recall"],
        )
    logger.info(separator)
    logger.info("--- End Training Summary ---")
```

### Loading Saved Checkpoint for Evaluation
```python
# Source: PyTorch saving/loading tutorial
model = MLP(input_dim=input_dim, hidden_layers=[128, 64, 32],
            num_classes=2, dropout=0.3)
model.load_state_dict(torch.load("outputs/checkpoints/best_model.pt",
                                  weights_only=True))
model.to(device)
model.eval()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torch.save(model, path) | torch.save(model.state_dict(), path) | PyTorch 1.x+ recommendation | Avoids pickle fragility; state_dict is just an OrderedDict |
| torch.load(path) | torch.load(path, weights_only=True) | PyTorch 2.0+ | Security: prevents arbitrary code execution from untrusted .pt files |
| Manual F1 calculation | sklearn.metrics.f1_score | Always preferred | Handles edge cases (zero division, empty classes) |
| BCELoss + sigmoid | CrossEntropyLoss with num_classes=2 | Standard practice | More numerically stable; works directly with class indices |

**Deprecated/outdated:**
- `torch.load(path)` without `weights_only=True`: Generates a FutureWarning in PyTorch >=2.4. Always pass `weights_only=True` for state_dicts.

## Open Questions

1. **Config validation for new keys**
   - What we know: config.py validates required nested keys per section. New keys `standalone_epochs` and `val_split` in the `training` section are optional with defaults.
   - What's unclear: Whether to add them to `_REQUIRED_NESTED` validation or keep them optional with `.get()` defaults.
   - Recommendation: Keep optional with `.get()` defaults. They are not required for FL training (Phase 3), only for standalone mode. Adding them to validation would break Phase 3 configs that don't include them.

2. **Gradient clipping**
   - What we know: User left this to Claude's discretion. CICIDS2017 features are already StandardScaler-normalized, reducing gradient explosion risk.
   - What's unclear: Whether gradient clipping adds measurable benefit for this specific dataset.
   - Recommendation: Skip gradient clipping for v1. The data is well-conditioned (normalized, no extreme outliers after preprocessing). If training becomes unstable, add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` as a follow-up.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0.0 (already in dev dependencies) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` (testpaths = ["tests"]) |
| Quick run command | `python -m pytest tests/test_model.py tests/test_train.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODL-01 | MLP instantiation with configurable layers, correct output shape | unit | `python -m pytest tests/test_model.py::TestMLP -x` | No -- Wave 0 |
| MODL-01 | Forward pass produces correct output dimensions | unit | `python -m pytest tests/test_model.py::TestMLP::test_forward_shape -x` | No -- Wave 0 |
| MODL-02 | train_one_epoch returns finite loss, model params change | unit | `python -m pytest tests/test_train.py::TestTrainOneEpoch -x` | No -- Wave 0 |
| MODL-02 | evaluate returns all required metrics (loss, acc, f1, prec, rec) | unit | `python -m pytest tests/test_train.py::TestEvaluate -x` | No -- Wave 0 |
| MODL-03 | Checkpoint saved when F1 improves, not saved when F1 doesn't improve | unit | `python -m pytest tests/test_train.py::TestCheckpointing -x` | No -- Wave 0 |
| MODL-03 | Saved checkpoint can be loaded and produces same predictions | unit | `python -m pytest tests/test_train.py::TestCheckpointing::test_load_checkpoint -x` | No -- Wave 0 |
| DATA-04 | Class weights loaded from JSON and applied to CrossEntropyLoss | unit | `python -m pytest tests/test_train.py::TestWeightedLoss -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_model.py tests/test_train.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_model.py` -- covers MODL-01 (MLP instantiation, forward pass, parameter count)
- [ ] `tests/test_train.py` -- covers MODL-02, MODL-03, DATA-04 (training loop, evaluation, checkpointing, weighted loss)
- [ ] Shared fixtures in `tests/conftest.py` -- add synthetic data fixtures for model/training tests (small tensors, mock class_weights.json)

## Sources

### Primary (HIGH confidence)
- Project codebase analysis: `src/federated_ids/config.py`, `seed.py`, `device.py`, `data/preprocess.py`, `data/partition.py` -- full API surface understood
- `config/default.yaml` -- exact config structure with model.hidden_layers=[128,64,32], model.dropout=0.3, training.learning_rate=0.001, model.num_classes=2
- `pyproject.toml` -- torch>=2.4.0, scikit-learn>=1.5.0, pytest>=8.0.0 confirmed
- [PyTorch CrossEntropyLoss docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) -- weight parameter API, raw logits requirement
- [PyTorch Saving/Loading Tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) -- state_dict pattern, weights_only=True
- [sklearn f1_score docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) -- average="binary" for binary classification

### Secondary (MEDIUM confidence)
- [PyTorch Forums: CrossEntropyLoss device placement](https://discuss.pytorch.org/t/correct-weights-assignment-for-crossentropyloss-and-cuda/29859) -- weight tensor must be on same device
- [PyTorch Forums: stratified split](https://discuss.pytorch.org/t/how-to-do-a-stratified-split/62290) -- random_split does not support stratification

### Tertiary (LOW confidence)
- CICIDS2017 MLP performance benchmarks (multiple papers) -- F1 > 0.96 reported consistently, but exact reproduction depends on feature selection and preprocessing; our 0.80 target is conservative and safe

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already pinned in pyproject.toml, APIs verified against official docs
- Architecture: HIGH - MLP pattern is straightforward nn.Module; train/evaluate separation is standard PyTorch practice; integration points with Phase 1 fully mapped from codebase analysis
- Pitfalls: HIGH - device placement, logit/softmax confusion, and eval mode toggle are well-documented PyTorch gotchas confirmed by multiple forum discussions
- Performance targets: HIGH - CICIDS2017 binary DDoS classification consistently achieves F1 > 0.96 in literature; target of 0.80 in 5 epochs is very conservative

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable domain -- PyTorch MLP patterns change slowly)
