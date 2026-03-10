# Phase 4: Evaluation and Visualization - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Evaluate the final federated model on held-out test data and produce publication-quality plots demonstrating the system works. Covers confusion matrix, classification report, convergence plots, per-client comparison, and TensorBoard integration. No new training capabilities or model changes.

</domain>

<decisions>
## Implementation Decisions

### Plot style and thesis formatting
- seaborn 'whitegrid' theme, muted color palette, clear axis labels, no decorative elements
- Figure size: 10x6 inches for all plots
- DPI: 150 for saved PNG files
- Save all plots to `outputs/plots/`

### Confusion matrix presentation
- Show both normalized percentages and absolute counts in cells (e.g. "95.2% (1423)")
- Blues colormap for heatmap
- Class labels: "BENIGN" / "DDoS" on axes
- Classification report: print to console AND save to `outputs/plots/classification_report.txt`

### Per-client comparison
- Grouped bar chart: one group per metric (F1, Precision, Recall), one bar per client
- Evaluate each client's locally-trained model AND the global model on the held-out test set
- This demonstrates federation benefit: global model should outperform any single local model

### Convergence plots
- Global metrics only (no per-client overlay)
- Two subplots side by side: loss over rounds + F1 over rounds
- Data sourced from per-round metrics already saved in `fl_metrics.json`

### TensorBoard integration
- Scalars only (loss, accuracy, F1, precision, recall per round)
- Integrated into FL loop: SummaryWriter logs inside `run_federated_training()` after each round
- Log files stored at `outputs/tensorboard/`
- Global metrics only (no per-client TensorBoard logging)
- `tensorboard` package needs to be added to pyproject.toml dependencies

### Claude's Discretion
- Exact seaborn/matplotlib styling details (font sizes, tick formatting, grid density)
- Loading skeleton for confusion matrix annotations
- How to structure the eval module internally (single file vs multiple)
- TensorBoard run naming convention

</decisions>

<specifics>
## Specific Ideas

- Convergence plot is the "hero figure" for the thesis — should clearly show F1 improving over FL rounds
- Per-client bar chart should make it visually obvious that federation produces a better model than any single client
- Classification report text file should be copy-pasteable into a LaTeX table

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluate()` in `model/train.py`: Returns `{loss, accuracy, f1, precision, recall}` with binary averaging — reuse for confusion matrix and per-client evaluation
- `save_fl_metrics()` in `fl/server.py`: Already saves per-round metrics as JSON to `outputs/metrics/fl_metrics.json` — convergence plots can read this
- `server_evaluate()` in `fl/server.py`: Loads global params into model and evaluates — reuse for held-out test set evaluation
- `MLP` model class in `model/model.py`: Needed to instantiate models for per-client local evaluation
- `partition_iid()` in `data/partition.py`: Needed to recreate client partitions for per-client evaluation

### Established Patterns
- Output directory: `./outputs` with subdirectories (`checkpoints/`, `metrics/`) — add `plots/` and `tensorboard/`
- Config-driven: all hyperparameters in `config/default.yaml`
- Logging via Python `logging` module, not print statements
- Global model checkpoint at `outputs/checkpoints/global_model.pt`

### Integration Points
- `run_federated_training()` in `fl/server.py`: TensorBoard SummaryWriter must be added here for live logging
- `fl_metrics.json`: Convergence plots read from this file (already has per-round loss, accuracy, f1, precision, recall)
- `pyproject.toml`: Add `tensorboard` dependency, add new console script entry point for evaluation

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-evaluation-and-visualization*
*Context gathered: 2026-03-09*
