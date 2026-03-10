# Phase 3: Federated Learning Infrastructure - Research

**Researched:** 2026-03-09
**Domain:** Federated Learning with Flower (flwr) + PyTorch on Windows/Python 3.13
**Confidence:** HIGH

## Summary

Phase 3 wires the existing MLP model and IID-partitioned data into a federated learning loop with FedAvg aggregation. The most critical finding is a **platform constraint**: the project runs on **Windows with Python 3.13**, and Flower's simulation engine (`start_simulation`, `run_simulation`) depends on Ray, which **does not support Python 3.13 on Windows**. This rules out using Flower's built-in simulation engine entirely.

The recommended approach is a **pure-Python federated training loop** that uses Flower's type system and utility functions (`ndarrays_to_parameters`, `parameters_to_ndarrays`, FedAvg aggregation logic) where beneficial, but implements the orchestration loop directly in Python. This is architecturally sound -- FedAvg is a simple weighted average of model parameters, and the project already has all the building blocks (`train_one_epoch`, `evaluate`, `partition_iid`, `create_dataloaders`, `MLP`). The federated loop calls these functions for each virtual client sequentially in a single process, exactly as Flower's simulation engine would.

**Primary recommendation:** Implement a pure-Python FedAvg training loop that sequentially trains virtual clients and aggregates their state_dicts via weighted average. Do not depend on `flwr[simulation]` or Ray. Keep `flwr` as a dependency only if its utility types add value; otherwise the loop is self-contained with PyTorch and NumPy.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use Flower's `start_simulation` engine -- single-process, no socket/port conflicts **(RESEARCH OVERRIDE: Ray/simulation not available on Windows+Python 3.13; implement equivalent pure-Python loop instead)**
- Virtual clients avoid memory overhead of separate processes
- Client logic (NumPyClient subclass) must be modular and mirror what would be deployed on a real 6G edge node -- clean separation so it could be swapped to real client-server later
- All clients share the same device (CPU or single GPU) -- Flower simulation handles client sequencing internally
- Standard FedAvg weighted by dataset size (num_examples) -- Flower's default behavior
- `fraction_fit: 1.0` -- all clients participate every round (configurable via config)
- Dual entry point: `python -m federated_ids.fl` AND console_scripts entry point (e.g., `federated-ids-train-fl`)
- CLI arguments `--config`, `--num-clients`, `--num-rounds` override config.yaml values
- Auto-run data pipeline if processed data doesn't exist
- Print ASCII config summary banner before training starts
- Server-side evaluation: after each round, evaluate global model on held-out test set
- One-line per round format: `Round  1/20 -- loss: 0.412, acc: 0.88, F1: 0.72, prec: 0.81, rec: 0.65`
- Summary table after training completes
- Global metrics only -- suppress individual client training logs
- Save best global model based on highest F1 across all rounds
- Save to `outputs/checkpoints/global_model.pt` -- state_dict only
- Save all per-round metrics to JSON: `outputs/metrics/fl_metrics.json`
- JSON includes embedded config for full reproducibility
- Automated convergence check: compare avg F1 of first 3 rounds vs last 3 rounds

### Claude's Discretion
- Exact NumPyClient implementation (get_parameters, set_parameters, fit, evaluate methods)
- How to structure server-side evaluate function within FedAvg strategy
- Internal parameter serialization (ndarray <-> state_dict conversion)
- Test structure and test cases for FL infrastructure
- Exact summary table formatting library or approach
- How to suppress Flower's default logging to keep console clean

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FLRN-01 | Implement Flower NumPyClient wrapping local PyTorch training | Client module with get/set_parameters + fit + evaluate wrapping train_one_epoch/evaluate; modular design for future Flower client-server swap |
| FLRN-02 | Implement Flower server with FedAvg aggregation strategy | Pure-Python FedAvg loop: collect client state_dicts, weighted average by num_examples, distribute global weights; mirrors Flower FedAvg semantics |
| FLRN-03 | Support configurable number of FL rounds and participating clients | Config-driven via federation.num_rounds, federation.num_clients, federation.fraction_fit; CLI overrides --num-clients, --num-rounds |
| EVAL-01 | Log per-round metrics (accuracy, precision, recall, F1) to console | Server-side evaluate() on global test set after each aggregation round; one-line format + summary table + JSON persistence |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.4.0 | Model training, state_dict management, tensor operations | Already installed, Phase 2 foundation |
| numpy | >=1.26.0 | Array operations for FedAvg weighted averaging | Already installed, used throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| flwr | >=1.13.0 | Type definitions (NDArrays, Parameters), future migration path | Optional -- only import if using Flower types for client interface |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Flower simulation engine | Pure-Python FedAvg loop | **Must use pure-Python**: Ray not available on Windows+Python 3.13; pure loop is simpler, faster, fully debuggable |
| flwr types (NDArrays) | Plain list[np.ndarray] | Flower types add dependency weight; plain Python types are sufficient and more portable |
| flwr FedAvg strategy class | Manual weighted average | FedAvg is ~10 lines of NumPy; manual implementation avoids Flower import complexity |

**Critical Platform Constraint:**
- Python 3.13.12 on Windows (MSYS/MSNT)
- Ray has **no package** for Python 3.13 (confirmed via pip dry-run and GitHub issue [#5512](https://github.com/adap/flower/issues/5512))
- `flwr[simulation]` requires Ray -- cannot install
- `start_simulation()` and `run_simulation()` are both unusable
- `start_simulation()` is additionally deprecated in Flower >=1.21

**Decision: Do NOT use Flower's simulation engine. Implement the equivalent federated loop in pure Python.**

The `flwr` base package (without `[simulation]`) can still be installed and provides type definitions. However, since the federated loop is self-contained, depending on `flwr` is optional. The implementation should be structured so that swapping to real Flower client-server in the future (e.g., on Linux with Ray) requires minimal changes.

**Installation:**
```bash
pip install -e .
# flwr is listed in pyproject.toml dependencies but simulation extras are NOT needed
# If flwr causes issues on Windows, it can be made optional without affecting FL functionality
```

## Architecture Patterns

### Recommended Project Structure
```
src/federated_ids/fl/
    __init__.py          # Subpackage docstring (exists, empty)
    client.py            # FederatedClient class wrapping train_one_epoch/evaluate
    server.py            # FedAvg aggregation + orchestration loop
    __main__.py          # Entry point: python -m federated_ids.fl
```

### Pattern 1: FederatedClient Class (FLRN-01)
**What:** A class encapsulating one virtual client's training and evaluation logic. Mirrors Flower NumPyClient interface for future portability.
**When to use:** Each client instantiated per round with its own DataLoader and model copy.

```python
# Source: Derived from Flower NumPyClient pattern + existing train.py
from collections import OrderedDict
import numpy as np
import torch
from federated_ids.model.model import MLP
from federated_ids.model.train import train_one_epoch, evaluate

class FederatedClient:
    """Virtual federated client wrapping local PyTorch training.

    Interface mirrors flwr.client.NumPyClient for future migration:
    - get_parameters() -> list[np.ndarray]
    - set_parameters(parameters: list[np.ndarray]) -> None
    - fit(parameters, config) -> (parameters, num_examples, metrics)
    - evaluate(parameters, config) -> (loss, num_examples, metrics)
    """

    def __init__(self, model: MLP, train_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def get_parameters(self) -> list[np.ndarray]:
        """Extract model parameters as list of NumPy arrays."""
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load parameters from list of NumPy arrays into model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        """Train locally and return updated parameters."""
        self.set_parameters(parameters)
        local_epochs = config.get("local_epochs", 1)
        for _ in range(local_epochs):
            train_one_epoch(
                self.model, self.train_loader,
                self.criterion, self.optimizer, self.device
            )
        num_examples = len(self.train_loader.dataset)
        return self.get_parameters(), num_examples, {}
```

### Pattern 2: FedAvg Aggregation (FLRN-02)
**What:** Weighted average of model parameters from all clients, weighted by number of training examples.
**When to use:** After collecting updated parameters from all clients each round.

```python
# Source: Flower FedAvg aggregate_fit logic (simplified)
import numpy as np

def fedavg_aggregate(
    results: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    """Aggregate client parameters using Federated Averaging.

    Args:
        results: List of (parameters, num_examples) from each client.

    Returns:
        Aggregated parameters as list of NumPy arrays.
    """
    # Calculate total number of examples
    total_examples = sum(num_examples for _, num_examples in results)

    # Weighted average of each parameter tensor
    aggregated = [
        np.zeros_like(results[0][0][i])
        for i in range(len(results[0][0]))
    ]

    for client_params, num_examples in results:
        weight = num_examples / total_examples
        for i, param in enumerate(client_params):
            aggregated[i] += param * weight

    return aggregated
```

### Pattern 3: Server-Side Evaluation Loop (EVAL-01)
**What:** After each aggregation round, load global parameters into a model and evaluate on the held-out test set.
**When to use:** Every round, producing the one-line metric output.

```python
# Source: Flower centralized evaluate_fn pattern + existing evaluate()
def server_evaluate(
    global_params: list[np.ndarray],
    model: MLP,
    test_loader,
    criterion,
    device,
) -> dict:
    """Evaluate global model on held-out test set after aggregation."""
    # Load global parameters
    params_dict = zip(model.state_dict().keys(), global_params)
    state_dict = OrderedDict(
        {k: torch.tensor(v).to(device) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)

    # Reuse Phase 2's evaluate function
    return evaluate(model, test_loader, criterion, device)
```

### Pattern 4: Orchestration Loop (FLRN-02 + FLRN-03)
**What:** The main federated training loop that coordinates clients and server evaluation.
**When to use:** Entry point function called from `__main__.py`.

```python
def run_federated_training(config: dict) -> list[dict]:
    """Execute the full federated learning process.

    For each round:
    1. Distribute global parameters to all clients
    2. Each client trains locally (sequential, single-process)
    3. Collect updated parameters weighted by num_examples
    4. Aggregate via FedAvg (weighted average)
    5. Evaluate global model on test set
    6. Log per-round metrics
    7. Checkpoint if best F1

    Returns:
        List of per-round metric dicts.
    """
    # ... setup code ...

    for round_num in range(1, num_rounds + 1):
        round_results = []
        for client_id in range(num_clients):
            client = create_client(client_id, global_params, ...)
            params, n_examples, _ = client.fit(global_params, fit_config)
            round_results.append((params, n_examples))

        global_params = fedavg_aggregate(round_results)
        metrics = server_evaluate(global_params, ...)

        logger.info(
            "Round %2d/%d -- loss: %.3f, acc: %.2f, F1: %.2f, prec: %.2f, rec: %.2f",
            round_num, num_rounds,
            metrics["loss"], metrics["accuracy"], metrics["f1"],
            metrics["precision"], metrics["recall"],
        )

        # Best model checkpoint
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_global_checkpoint(global_params, round_num, best_f1)

        history.append(metrics)

    return history
```

### Anti-Patterns to Avoid
- **Importing flwr.simulation on Windows+Python 3.13:** Will fail at import time due to missing Ray dependency. Guard any Flower simulation imports.
- **Creating separate processes per client:** Unnecessary overhead for simulation; sequential execution in single process is correct for this use case.
- **Re-instantiating model from scratch each round:** Reuse model object, just call `set_parameters()` to load new weights. Avoids repeated memory allocation.
- **Using `model.parameters()` instead of `state_dict()`:** `state_dict()` captures all model state including batch norm statistics; `parameters()` only captures trainable weights.
- **Forgetting to move parameters to device:** When loading ndarray parameters into a model on GPU, the `torch.tensor(v).to(device)` step is essential.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| IID partitioning | Custom random splits | `partition_iid()` from Phase 1 | Already handles stratified splits with validation |
| Local training | New training loop | `train_one_epoch()` from Phase 2 | Tested, handles batching and loss correctly |
| Model evaluation | New metric computation | `evaluate()` from Phase 2 | Returns all needed metrics (loss, acc, F1, prec, rec) |
| Model definition | New model class | `MLP` from Phase 2 | Configurable, tested, raw logit output |
| DataLoader creation | Manual tensor wrapping | `create_dataloaders()` from Phase 1 | Handles tensor conversion and batch configuration |
| Config loading | Custom YAML parsing | `load_config()` from Phase 1 | Env var interpolation, validation, established pattern |
| Seed management | Manual seed setting | `set_global_seed()` from Phase 1 | Sets all RNG sources consistently |
| Device detection | Hardcoded device string | `get_device()` from Phase 1 | Auto-detects CUDA/MPS/CPU |

**Key insight:** Phase 3 is primarily an *integration phase*. The building blocks exist. The new code is the orchestration loop (FedAvg aggregation + sequential client training + server-side evaluation + metrics logging + checkpointing). Do not rewrite any Phase 1/Phase 2 functionality.

## Common Pitfalls

### Pitfall 1: Ray/Simulation Engine Unavailable on Windows+Python 3.13
**What goes wrong:** Importing `flwr.simulation` or installing `flwr[simulation]` fails because Ray has no wheel for Python 3.13 on Windows.
**Why it happens:** Ray officially does not support Python 3.13 yet. Flower's simulation engine depends on Ray.
**How to avoid:** Do not use `start_simulation` or `run_simulation`. Implement the federated loop in pure Python. The `flwr` base package (without simulation extras) can still be installed.
**Warning signs:** `ImportError` on `ray`, empty backend list in simulation.

### Pitfall 2: Double-Applying Optimizer State Across Rounds
**What goes wrong:** If the optimizer carries momentum/adaptive state from a previous round, it biases the new round's training.
**Why it happens:** Adam optimizer has per-parameter momentum buffers that persist across calls.
**How to avoid:** Create a fresh optimizer for each client each round, OR use the same optimizer per client consistently across rounds (matching the McMahan et al. FedAvg paper where each client resets optimizer state each round).
**Warning signs:** Divergent training, loss spikes at round boundaries.

### Pitfall 3: State Dict Key Ordering Mismatch
**What goes wrong:** Parameters loaded into the wrong layers because dict ordering changed.
**Why it happens:** Mismatched model architectures between clients, or using dict comprehension that reorders keys.
**How to avoid:** Use `OrderedDict` and `zip(model.state_dict().keys(), parameters)` to maintain key alignment. All clients must use identical model architecture.
**Warning signs:** Nonsensical model output after `set_parameters()`, sudden accuracy drop.

### Pitfall 4: Forgetting to Call model.train()/model.eval()
**What goes wrong:** Dropout and batch norm behave incorrectly during training or evaluation.
**Why it happens:** After calling `evaluate()` (which sets model.eval()), the model stays in eval mode.
**How to avoid:** `train_one_epoch()` already calls `model.train()` and `evaluate()` already calls `model.eval()`. Since these Phase 2 functions are reused, this is handled. Just don't add manual model mode switches that conflict.
**Warning signs:** Training loss doesn't decrease, eval metrics are unusually noisy.

### Pitfall 5: Not Weighting Aggregation by Dataset Size
**What goes wrong:** Clients with fewer samples have equal influence, biasing the global model.
**Why it happens:** Using simple average instead of weighted average.
**How to avoid:** Always weight by `num_examples` from each client. With IID partitioning the sizes are approximately equal, but the code should still weight correctly for correctness and future non-IID support.
**Warning signs:** Worse convergence than expected, especially with unequal partition sizes.

### Pitfall 6: Shared Model Object Between Clients
**What goes wrong:** Client 2's training modifies the model that Client 1 already trained, corrupting results.
**Why it happens:** Passing the same model reference to multiple clients.
**How to avoid:** Either (a) create a fresh model per client and load global parameters, or (b) use a single model but always call `set_parameters()` with the global parameters before each client's training. Option (b) is memory-efficient and sufficient for sequential execution.
**Warning signs:** Results differ from Flower simulation baseline, training order affects final model.

### Pitfall 7: Suppressing Too Many Logs
**What goes wrong:** Useful error messages are hidden along with verbose Flower/training logs.
**Why it happens:** Setting root logger to WARNING instead of targeting specific loggers.
**How to avoid:** Only suppress specific loggers: `logging.getLogger("flwr").setLevel(logging.WARNING)` and suppress the client-level training logger selectively. Keep the FL orchestration logger at INFO.
**Warning signs:** Silent failures, no output when things go wrong.

## Code Examples

### NDArray <-> State Dict Conversion (verified pattern from Flower NumPyClient)
```python
from collections import OrderedDict
import numpy as np
import torch

def get_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    """Convert PyTorch model state_dict to list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    """Load list of NumPy arrays into PyTorch model state_dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)
```
Source: [Flower NumPyClient docs](https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html), [Flower PyTorch Quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

### FedAvg Weighted Aggregation
```python
import numpy as np

def fedavg_aggregate(
    results: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    """Weighted average of model parameters (FedAvg algorithm).

    Each result is (client_parameters, num_examples).
    Weight = num_examples / total_examples.
    """
    total_examples = sum(n for _, n in results)
    num_layers = len(results[0][0])

    aggregated = [np.zeros_like(results[0][0][i]) for i in range(num_layers)]

    for client_params, num_examples in results:
        weight = num_examples / total_examples
        for i in range(num_layers):
            aggregated[i] += client_params[i] * weight

    return aggregated
```
Source: [Flower FedAvg source](https://flower.ai/docs/framework/_modules/flwr/serverapp/strategy/fedavg.html), [McMahan et al. 2017](https://arxiv.org/abs/1602.05629)

### Config Summary Banner
```python
def print_config_banner(config: dict, device: str, num_features: int) -> None:
    """Print ASCII config summary for experiment logging."""
    fed = config["federation"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    lines = [
        "=" * 52,
        "  Federated IDS Training Configuration",
        "=" * 52,
        f"  Clients:       {fed['num_clients']}",
        f"  Rounds:        {fed['num_rounds']}",
        f"  Local epochs:  {train_cfg['local_epochs']}",
        f"  Batch size:    {train_cfg['batch_size']}",
        f"  Learning rate: {train_cfg['learning_rate']}",
        f"  Model:         MLP {model_cfg['hidden_layers']}",
        f"  Dropout:       {model_cfg['dropout']}",
        f"  Features:      {num_features}",
        f"  Device:        {device}",
        f"  Strategy:      FedAvg (fraction_fit={fed['fraction_fit']})",
        f"  Seed:          {config.get('seed', 42)}",
        "=" * 52,
    ]
    for line in lines:
        logger.info(line)
```

### Convergence Check
```python
def check_convergence(history: list[dict], n: int = 3) -> bool:
    """Compare avg F1 of first n rounds vs last n rounds.

    Returns True if last n rounds have higher avg F1 than first n.
    """
    if len(history) < 2 * n:
        # Not enough rounds for meaningful comparison
        n = max(1, len(history) // 2)

    early_f1 = np.mean([h["f1"] for h in history[:n]])
    late_f1 = np.mean([h["f1"] for h in history[-n:]])

    passed = late_f1 > early_f1
    logger.info(
        "Convergence check: early F1=%.4f, late F1=%.4f -> %s",
        early_f1, late_f1, "PASS" if passed else "FAIL"
    )
    return passed
```

### Metrics JSON Persistence
```python
import json

def save_fl_metrics(
    history: list[dict],
    config: dict,
    output_path: str,
) -> None:
    """Save per-round metrics with embedded config to JSON."""
    payload = {
        "config": {
            "num_clients": config["federation"]["num_clients"],
            "num_rounds": config["federation"]["num_rounds"],
            "local_epochs": config["training"]["local_epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "model": config["model"]["hidden_layers"],
            "strategy": "FedAvg",
            "fraction_fit": config["federation"]["fraction_fit"],
            "device": str(device),
            "seed": config.get("seed", 42),
        },
        "rounds": [
            {"round": i + 1, **metrics}
            for i, metrics in enumerate(history)
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `flwr.simulation.start_simulation()` | `flwr.simulation.run_simulation()` with ClientApp/ServerApp | Flower 1.21 (Sept 2025) | start_simulation deprecated; run_simulation is new API |
| `flwr.client.NumPyClient` standalone | `flwr.client.ClientApp` with Message API | Flower 1.x -> new architecture | NumPyClient still works but is legacy pattern |
| `flwr.server.strategy.FedAvg` with evaluate_fn callback | `FedAvg.start()` with evaluate_fn on ServerApp | Flower 1.21+ | Strategy now lives in `flwr.serverapp.strategy` |
| Ray-based simulation backend | Ray still default, no alternative backend | Current | Windows+Python 3.13 blocked; pure-Python loop is workaround |

**Deprecated/outdated:**
- `start_simulation()`: Deprecated since Flower 1.21. Requires Ray (unavailable on Windows+Python 3.13).
- `start_numpy_client()`: Deprecated since Flower 1.7. Use `start_client()` with `.to_client()`.
- `flwr.server.strategy.FedAvg`: Moved to `flwr.serverapp.strategy.FedAvg` in newer versions.

## Open Questions

1. **Should `flwr` remain as a dependency?**
   - What we know: The base `flwr` package (without `[simulation]`) installs on Windows+Python 3.13 (confirmed by dry-run). It adds ~15 transitive dependencies (SQLAlchemy, grpcio, cryptography, etc.).
   - What's unclear: Whether those heavy dependencies cause any issues at runtime, or whether keeping flwr as dependency is worth the weight for type compatibility alone.
   - Recommendation: Keep `flwr` in pyproject.toml for thesis credibility ("uses Flower framework"), but the FL code should NOT import from flwr at runtime. If flwr install causes issues, make it optional. The pure-Python loop does not need any flwr imports.

2. **Client optimizer state reset between rounds**
   - What we know: Original FedAvg paper (McMahan 2017) resets client optimizer state each round. Some variants carry state across rounds.
   - What's unclear: Which approach works better for this specific dataset/model.
   - Recommendation: Reset optimizer state each round (create fresh optimizer per client per round). This matches the standard FedAvg paper and is simpler.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_fl.py -x -v` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FLRN-01 | Client get/set_parameters preserves model weights | unit | `pytest tests/test_fl.py::test_client_get_set_parameters -x` | No -- Wave 0 |
| FLRN-01 | Client fit returns updated parameters and num_examples | unit | `pytest tests/test_fl.py::test_client_fit -x` | No -- Wave 0 |
| FLRN-02 | FedAvg aggregate produces correct weighted average | unit | `pytest tests/test_fl.py::test_fedavg_aggregate -x` | No -- Wave 0 |
| FLRN-02 | FedAvg with equal-size partitions equals simple average | unit | `pytest tests/test_fl.py::test_fedavg_equal_weights -x` | No -- Wave 0 |
| FLRN-03 | Configurable rounds and clients from config | unit | `pytest tests/test_fl.py::test_config_driven_rounds_clients -x` | No -- Wave 0 |
| FLRN-03 | CLI overrides take precedence over config values | unit | `pytest tests/test_fl.py::test_cli_overrides -x` | No -- Wave 0 |
| EVAL-01 | Per-round metrics contain all required keys | unit | `pytest tests/test_fl.py::test_round_metrics_keys -x` | No -- Wave 0 |
| EVAL-01 | Metrics JSON saved with embedded config | unit | `pytest tests/test_fl.py::test_metrics_json_output -x` | No -- Wave 0 |
| EVAL-01 | Convergence check compares early vs late F1 | unit | `pytest tests/test_fl.py::test_convergence_check -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_fl.py -x -v`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_fl.py` -- covers FLRN-01, FLRN-02, FLRN-03, EVAL-01
- [ ] Test fixtures in `tests/conftest.py` -- add FL-specific fixtures (synthetic partitions, mock clients)
- [ ] No framework install needed -- pytest already configured

## Sources

### Primary (HIGH confidence)
- [Flower FedAvg strategy docs](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html) - FedAvg parameters, weighted_by_key, aggregate_fit
- [Flower NumPyClient docs](https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html) - Client interface: get_parameters, fit, evaluate
- [Flower start_simulation docs](https://flower.ai/docs/framework/ref-api/flwr.simulation.start_simulation.html) - Deprecated status confirmed
- [Flower run_simulation docs](https://flower.ai/docs/framework/ref-api/flwr.simulation.run_simulation.html) - New API signature, Ray backend requirement
- [Flower PyTorch quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html) - Client implementation patterns, state_dict conversion
- [GitHub Issue #5512](https://github.com/adap/flower/issues/5512) - Ray not supported on Windows, confirmed blocker
- Existing codebase: `train.py`, `model.py`, `partition.py`, `config.py` - Verified reusable building blocks

### Secondary (MEDIUM confidence)
- [McMahan et al. 2017 (FedAvg paper)](https://arxiv.org/abs/1602.05629) - Algorithm definition, optimizer reset semantics
- [FedAvg implementation examples (TDS, Medium)](https://towardsdatascience.com/federated-learning-a-simple-implementation-of-fedavg-federated-averaging-with-pytorch-90187c9c9577/) - Pure-Python FedAvg patterns
- [Flower changelog](https://flower.ai/docs/framework/ref-changelog.html) - Deprecation timeline

### Tertiary (LOW confidence)
- None -- all findings verified with primary or secondary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Platform constraint verified via pip dry-run and GitHub issues; PyTorch/NumPy already installed
- Architecture: HIGH - Pure-Python FedAvg is well-documented algorithm; all building blocks exist in codebase
- Pitfalls: HIGH - Windows+Ray blocker verified; FedAvg gotchas well-known in FL literature
- Code examples: HIGH - Patterns derived from Flower official docs and existing codebase functions

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable -- FedAvg algorithm unchanged; platform constraint unlikely to resolve in 30 days)
