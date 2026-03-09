# Pitfalls Research

**Domain:** Federated Learning Intrusion Detection System (FL-IDS) for 6G Edge Networks
**Researched:** 2026-03-09
**Confidence:** HIGH (CICIDS2017 issues, evaluation methodology), MEDIUM (Flower API specifics)

**Note on sources:** Web search tools were unavailable during this research. Findings are based on training data covering extensive academic literature on FL-IDS systems, CICIDS2017 dataset analyses, and Flower framework documentation through early 2025. CICIDS2017 pitfalls and FL evaluation methodology issues are among the most-discussed topics in network security ML literature, giving HIGH confidence despite reliance on training data. Flower API details should be verified against current docs during implementation.

---

## Critical Pitfalls

### Pitfall 1: CICIDS2017 Contains Infinity and NaN Values That Crash Training

**What goes wrong:**
The CICIDS2017 CSV files contain `Inf`, `-Inf`, and `NaN` values in several numeric columns, most notably `Flow Bytes/s` and `Flow Packets/s`. Loading the CSVs with Pandas and feeding them into a PyTorch model without cleaning causes immediate NaN loss, silent model corruption, or outright crashes. This is the single most common first-time failure with CICIDS2017.

**Why it happens:**
The CICFlowMeter tool that generated the dataset divides by zero when computing per-second flow statistics for flows with zero duration. The resulting infinity values propagate through normalization and into model weights. Pandas reads these as `float('inf')` which NumPy and PyTorch propagate silently through computations until the loss becomes NaN.

**How to avoid:**
1. After loading each CSV, immediately run: `df.replace([np.inf, -np.inf], np.nan, inplace=True)` followed by `df.dropna()` (or impute with column median).
2. Assert no infinities or NaNs remain before any further processing: `assert not df.isin([np.inf, -np.inf]).any().any()` and `assert not df.isna().any().any()`.
3. Do this BEFORE train/test split, BEFORE normalization, BEFORE any feature engineering.

**Warning signs:**
- Loss becomes `NaN` after first forward pass
- Model outputs all-same predictions (all 0 or all 1)
- Scikit-learn `StandardScaler` produces `NaN` or warnings about invalid values
- Accuracy suspiciously at 0.0 or 50.0% from the start

**Phase to address:** Data preprocessing (Phase 1). Must be the very first step after CSV loading.

---

### Pitfall 2: CICIDS2017 Has Extreme Class Imbalance That Inflates Accuracy

**What goes wrong:**
CICIDS2017 contains approximately 2.8 million flows, of which roughly 80% are benign traffic. For DDoS specifically, attack flows are a small fraction of the total. A model that predicts "benign" for every sample achieves ~80%+ accuracy, making accuracy a misleading metric. Teams report "95% accuracy" without realizing their model barely detects attacks.

**Why it happens:**
The dataset reflects realistic traffic distribution where attacks are minority events. Standard accuracy weighting treats each sample equally, so the dominant class drives the metric. Cross-entropy loss also biases toward the majority class unless corrected.

**How to avoid:**
1. NEVER use accuracy as the primary metric. Use precision, recall, and F1-score (macro or weighted), and always report the confusion matrix.
2. For binary classification (benign vs DDoS), report attack-class recall specifically -- a model with 99% accuracy but 30% DDoS recall is useless as an IDS.
3. Apply one or more balancing strategies:
   - Undersample benign traffic to match DDoS volume (simplest, loses data)
   - Use class weights in PyTorch loss: `nn.CrossEntropyLoss(weight=class_weights)` where weights are inversely proportional to class frequency
   - SMOTE is an option but adds complexity and is harder to justify in FL contexts
4. When splitting data across FL clients, ensure each client has a similar class distribution -- random splitting can give one client almost no attack samples.

**Warning signs:**
- Accuracy is very high (>95%) but recall for attack class is low (<70%)
- Confusion matrix shows almost all predictions in one class
- Loss converges very quickly (model learned the majority shortcut)
- F1-macro is significantly lower than F1-weighted

**Phase to address:** Data preprocessing (Phase 1) for balancing strategy; Evaluation (final phase) for metric selection. Must decide the balancing approach before federated partitioning.

---

### Pitfall 3: Data Leakage Through Naive Train/Test Splitting

**What goes wrong:**
CICIDS2017 flows are temporally ordered and contain correlated features within sessions. Randomly shuffling all data and splitting produces train/test sets that contain flows from the same TCP sessions, same source-destination pairs, and same time windows. The model memorizes session-specific patterns rather than learning generalizable attack signatures. Reported metrics are inflated by 5-20% compared to honest evaluation.

**Why it happens:**
Standard `train_test_split(shuffle=True)` intermixes flows from the same network sessions. Features like source/destination IP, port numbers, and flow identifiers create trivial shortcuts. The model can match test flows to training flows from the same session without learning actual attack behavior.

**How to avoid:**
1. Remove or do not use identifying features: `Source IP`, `Destination IP`, `Source Port`, `Destination Port`, `Flow ID`, `Timestamp`. These are leakage vectors, not generalizable features.
2. If doing temporal evaluation, split by time (earlier data for training, later for testing) rather than random shuffle.
3. For this project (IID FL simulation), removing the identifier columns and then shuffling is acceptable, but document the decision.
4. Apply normalization (StandardScaler/MinMaxScaler) fitted ONLY on training data, then transform test data. Fitting on the full dataset is a subtler form of leakage.

**Warning signs:**
- Test performance is suspiciously close to training performance (overfit looks like generalization)
- Model degrades significantly when evaluated on a completely different day's data from CICIDS2017
- Feature importance analysis shows IP addresses or ports as top features

**Phase to address:** Data preprocessing (Phase 1). Must be decided before partitioning data across clients.

---

### Pitfall 4: Flower Client-Server Parameter Shape Mismatch

**What goes wrong:**
The Flower client's `get_parameters()` and `set_parameters()` methods must return and accept model parameters in exactly the same order and shape as the server expects. If the PyTorch model definition differs even slightly between client and server (different layer order, extra bias term, different initialization), FedAvg produces a corrupted global model. The model appears to train but produces garbage predictions after aggregation.

**Why it happens:**
PyTorch's `state_dict()` ordering depends on the order layers are defined in `__init__`. If one client defines layers in a different order, or uses a slightly different architecture, the parameter lists have the same total number of floats but mapped to wrong layers. Flower does not validate parameter semantics -- it trusts that all clients share the same architecture.

**How to avoid:**
1. Define the model class in ONE shared file (e.g., `model.py`) imported by both `server.py` and `client.py`. Never duplicate the model definition.
2. Use a helper function for parameter conversion:
   ```python
   def get_parameters(model):
       return [val.cpu().numpy() for _, val in model.state_dict().items()]

   def set_parameters(model, parameters):
       params_dict = zip(model.state_dict().keys(), parameters)
       state_dict = {k: torch.tensor(v) for k, v in params_dict}
       model.load_state_dict(state_dict, strict=True)
   ```
3. The `strict=True` flag in `load_state_dict` catches shape mismatches early.
4. Log parameter shapes on both client and server side in the first round to verify alignment.

**Warning signs:**
- Global model accuracy drops sharply after aggregation despite good local training
- Different clients report wildly different metrics after receiving the same global model
- `RuntimeError` about tensor shape mismatches (the lucky case -- you get an error)
- Model works with 1 client but breaks with 2+

**Phase to address:** FL infrastructure setup (Phase 2). Must be established as an architectural pattern from the start.

---

### Pitfall 5: Evaluating Federated Model on Training Client Data Instead of Held-Out Data

**What goes wrong:**
Each FL client trains on its local partition. If the "evaluation" step tests the global model on the same data the clients trained on, the reported metrics reflect memorization, not generalization. This is especially insidious because FL tutorials often show local evaluation on the client's own data as the primary evaluation method.

**Why it happens:**
Flower's `evaluate()` method on the client is called after each round, and beginners use the client's training data for this evaluation. Since FedAvg averages weights from models that all saw their local data, the global model retains much of the local memorization. Additionally, with only 2-5 clients in IID setting, the data overlap after aggregation is high.

**How to avoid:**
1. Each client must hold out a LOCAL test set that is NEVER used for training. Split each client's partition into train (80%) and test (20%) before any training begins.
2. Additionally, hold out a GLOBAL test set that no client ever sees during training. Use this for final model evaluation.
3. Flower's `evaluate()` client method should use the local test set, not training data.
4. Final evaluation should use the centralized global test set against the final aggregated model.
5. Partition strategy: from the full preprocessed dataset, hold out 15-20% as global test. Split the remaining 80-85% across clients. Each client further splits into local train/test.

**Warning signs:**
- Federated metrics per round look nearly identical to centralized training metrics
- No gap between training and evaluation metrics (too good to be true)
- Model degrades noticeably on truly unseen data

**Phase to address:** Data partitioning (Phase 1) and FL evaluation pipeline (Phase 3).

---

### Pitfall 6: CICIDS2017 Column Name Inconsistencies Across CSV Files

**What goes wrong:**
The CICIDS2017 dataset is distributed as multiple CSV files (one per day: Monday, Tuesday, etc.). These files have inconsistent column names -- some have leading/trailing whitespace in column headers (e.g., ` Label` vs `Label`, ` Destination Port` vs `Destination Port`). Merging the files without cleaning column names causes duplicate columns, KeyError exceptions, or silently misaligned data.

**Why it happens:**
The CICFlowMeter tool that generated the CSVs had inconsistent output formatting across runs. Different daily capture files were processed separately. The Label column in particular often has a leading space: ` Label` instead of `Label`.

**How to avoid:**
1. After loading each CSV, strip column names: `df.columns = df.columns.str.strip()`
2. Standardize label values: strip whitespace from the Label column values too: `df['Label'] = df['Label'].str.strip()`
3. Verify column count and names match across all loaded files before concatenation.
4. Load and process each CSV individually, apply cleaning, then concatenate.

**Warning signs:**
- `KeyError: 'Label'` when the column is visually present
- DataFrame has 79 columns from one file and 78 from another
- Duplicate column names after merge (Pandas silently allows this)
- Label encoding produces unexpected category counts

**Phase to address:** Data preprocessing (Phase 1). Very first step alongside Inf/NaN cleaning.

---

### Pitfall 7: Non-Convergence Due to Mismatched Learning Rates in FL

**What goes wrong:**
The local learning rate (used by each client's SGD/Adam optimizer) and the number of local epochs interact with FedAvg in ways that do not match centralized training intuition. A learning rate that works well for centralized training (e.g., `lr=0.001` with Adam) can cause divergence in FL because each client drifts far from the global model in each round, and averaging drifted models produces a worse global model than any individual client had.

**Why it happens:**
In FedAvg, each client trains for E local epochs before sending weights back. If E is large or lr is high, clients "overfit" to their local distribution. The aggregated model ends up at a point in parameter space that is not good for any client's data. This is called "client drift." Even with IID data, this happens if local training is too aggressive.

**How to avoid:**
1. Start conservative: 1 local epoch, `lr=0.01` for SGD or `lr=0.001` for Adam.
2. Use more FL rounds with fewer local epochs rather than fewer rounds with many local epochs. Start with `E=1, rounds=50` and tune from there.
3. Monitor per-round global evaluation loss. It should decrease (not monotonically, but trending down). If it oscillates wildly or increases, reduce lr or local epochs.
4. For this project's IID case, client drift is less severe, but still possible with aggressive settings.
5. Log individual client losses alongside aggregated metrics to detect if one client is diverging.

**Warning signs:**
- Global loss oscillates wildly round-over-round
- Increasing number of FL rounds does not improve accuracy
- Individual client metrics are fine but aggregated model is worse than any client
- Accuracy drops sharply after aggregation step each round

**Phase to address:** FL training configuration (Phase 2-3). Hyperparameter tuning after basic pipeline works.

---

### Pitfall 8: Using the Wrong CICIDS2017 Files or Missing DDoS Data

**What goes wrong:**
CICIDS2017 is spread across multiple CSV files by day of the week. DDoS attacks are specifically in the Friday dataset (`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`). Teams sometimes load only the Monday (all-benign) file or miss the DDoS-specific file, ending up with a dataset that has zero or minimal attack samples for their target class.

**Why it happens:**
The dataset download provides 8 CSV files. The file naming is not intuitive. Some download mirrors only include partial files. The DDoS label appears only in Friday afternoon data. Other attack types (PortScan, Brute Force, etc.) are in other files, which the project explicitly scopes out.

**How to avoid:**
1. Document exactly which files are needed: for DDoS detection, at minimum the Friday DDoS file plus benign traffic from Monday (or other days for benign variety).
2. After loading, verify label distribution: `df['Label'].value_counts()` should show both BENIGN and DDoS counts.
3. Decide upfront whether to use only Friday data (DDoS + benign) or combine with other days' benign traffic for more variety.
4. Recommended for this project: use Friday DDoS file (has both benign and DDoS traffic from that session) supplemented with Monday benign traffic for volume. This gives a clean binary classification task.

**Warning signs:**
- Label value counts show zero DDoS samples
- Only one class present after filtering
- Model trivially achieves 100% accuracy (only one class in data)

**Phase to address:** Data acquisition/preprocessing (Phase 1). Must verify before any pipeline work.

---

### Pitfall 9: Flower API Version Incompatibility

**What goes wrong:**
Flower (flwr) has undergone significant API changes between versions. Code written for Flower 0.x or early 1.x does not work with current versions. Tutorials and StackOverflow answers often reference deprecated APIs (`start_server`, `start_numpy_client`, `NumPyClient`). Using outdated patterns causes import errors or subtle behavioral differences.

**Why it happens:**
Flower has been actively evolving. The shift from `NumPyClient` to the newer `Client`/`ClientApp` API, and from `start_server`/`start_client` to `ServerApp`/`ClientApp` with `flower-superlink`/`flower-superexec` architecture represents a major redesign. Most tutorials and academic papers reference the older API.

**How to avoid:**
1. Pin the Flower version in `requirements.txt` and verify all code examples against that specific version's documentation.
2. For a v1 proof-of-concept, the "classic" Flower API (`fl.client.NumPyClient`, `fl.server.start_server`) still works in recent versions and is simpler. The newer `ClientApp`/`ServerApp` API is more powerful but adds complexity.
3. Verify against official Flower docs at the exact pinned version, not against tutorials or blog posts.
4. If using Flower >= 1.5, check whether `start_numpy_client` has been renamed or deprecated -- the function signatures change across minor versions.

**Warning signs:**
- `ImportError` or `AttributeError` on Flower classes/functions
- Deprecation warnings at startup
- `start_server` or `start_client` behaving differently than tutorial shows
- Client connects but never receives model parameters

**Phase to address:** Project setup (Phase 1). Pin version and verify API before writing FL code.

---

### Pitfall 10: Feature Scaling Inconsistency Across FL Clients

**What goes wrong:**
Each FL client independently normalizes/scales its local data partition. Because different partitions have different min/max values and different means/standard deviations, the same raw feature value gets mapped to different scaled values on different clients. The model learns inconsistent feature representations, and averaging the weights produces a model that works poorly for everyone.

**Why it happens:**
With IID data, this effect is mitigated but not eliminated (especially with small client counts like 2-5 where statistical variation between partitions is significant). Each client fitting its own `StandardScaler` means feature X=100 might map to 0.5 on client A but 0.7 on client B.

**How to avoid:**
1. **Best approach:** Compute global normalization statistics (mean, std or min, max) from the training data BEFORE partitioning across clients. Share these statistics with all clients. Each client uses the SAME scaler parameters.
2. This does not leak individual data points -- aggregate statistics are privacy-safe for this project's scope.
3. Alternatively, use `MinMaxScaler` with explicitly set feature ranges based on the dataset's known feature ranges (more robust to partition differences).
4. Save the scaler object (pickle) so the final evaluation uses the same transformation.

**Warning signs:**
- Good per-client local metrics but poor global model metrics
- Different clients report very different feature value ranges in logs
- Model performance varies significantly by client after receiving global model

**Phase to address:** Data preprocessing (Phase 1). Must decide on normalization strategy before client partitioning.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoded file paths for CICIDS2017 CSVs | Quick data loading | Breaks on any other machine, impossible to reconfigure | Never -- use config or CLI args from the start |
| Skipping data validation assertions | Faster preprocessing code | Silent data corruption propagates to model; debugging takes hours | Never -- assertions cost nothing |
| Copy-pasting model definition into client.py and server.py | Fewer imports to manage | Inevitable divergence between files; aggregation corruption | Never -- single model.py from day one |
| Using accuracy as the only metric | Simple to compute and explain | Misleading with imbalanced data; masks poor attack detection | Never for IDS -- always include F1, precision, recall |
| Training all preprocessing in a Jupyter notebook | Fast exploration | Cannot integrate into FL pipeline; state leaks between cells; not reproducible | Only for initial data exploration; must port to .py scripts |
| Not setting random seeds | Fewer lines of code | Irreproducible results; cannot debug stochastic failures | Never -- set seeds for PyTorch, NumPy, Python random from the start |
| Using `torch.save(model)` instead of `state_dict()` | Simpler save/load | Pickle serialization issues; version-dependent; breaks with Flower param exchange | Never in FL context -- always use state_dict |
| Fitting scaler per-client independently | No coordination needed | Inconsistent feature spaces across clients | Only if proven that IID split makes this negligible (verify empirically) |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading all CICIDS2017 CSVs into memory at once | RAM exhaustion (>8GB), system swap thrashing, OOM kills | Load only needed files; drop unnecessary columns early; use appropriate dtypes (`float32` not `float64`) | Machines with <16GB RAM; all CICIDS2017 CSVs together are ~2GB+ in memory |
| Recalculating preprocessing every run | Minutes wasted on each experiment iteration | Save preprocessed data to parquet/pickle after initial cleaning; load cached version in FL pipeline | After first successful preprocessing |
| Using `float64` throughout PyTorch pipeline | 2x memory usage, slower training, no accuracy benefit for this task | Convert to `float32` after preprocessing: `tensor = torch.tensor(data, dtype=torch.float32)` | Always wasteful for MLP-based IDS |
| Too many FL rounds with no early stopping | Hours of training with no improvement after round 20 | Implement simple early stopping: if global eval metric has not improved in N rounds, stop | When FL rounds exceed 30-50 for this dataset/model |
| Sending full model every round (even when converged) | Unnecessary network overhead in simulation; slow round completion | For v1 simulation this is acceptable; for production, use delta compression or gradient sparsification | Only matters at real deployment scale |
| Large batch sizes on MLP with small per-client datasets | Fewer gradient updates per epoch, slower convergence | Start with batch_size=64 or 128; each client's partition may only have 50K-200K samples | When per-client data is small (<50K samples) |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Logging raw data samples in training output | Data that is supposed to stay private appears in logs | Log only aggregate statistics (mean, count, class distribution), never individual samples |
| Transmitting data statistics (min, max, mean) alongside model weights | Leaks information about local data distribution beyond what model weights already reveal | For v1, sharing global scaler stats pre-computed before partitioning is fine; do not share per-client stats post-partitioning |
| No validation of received model parameters | Malicious/corrupted client can poison the global model | Out of scope for v1 per PROJECT.md, but add TODO for v2 robust aggregation |
| Hardcoded paths that include usernames or machine-specific info | Reveals system information in committed code | Use relative paths, environment variables, or config files |
| Committing the CICIDS2017 dataset to git | Repository bloat (>1GB); dataset license concerns | Add `*.csv` and `data/` to `.gitignore`; document download instructions in README |

---

## "Looks Done But Isn't" Checklist

- [ ] **"Model trains successfully"** -- but does it train on properly cleaned data? Check for Inf/NaN in raw features.
- [ ] **"95% accuracy achieved"** -- but what is the attack-class recall? Check confusion matrix. A majority-class-only predictor hits >80%.
- [ ] **"FL training converges"** -- but is the global model evaluated on held-out data, or on client training data? Check evaluation pipeline.
- [ ] **"Clients communicate with server"** -- but are the model parameters correctly aligned? Check parameter shapes round-trip.
- [ ] **"Preprocessing pipeline works"** -- but is the scaler fitted only on training data? Check for train/test leakage.
- [ ] **"Data is split across clients"** -- but does each client have both classes? Check per-client label distribution.
- [ ] **"Confusion matrix looks good"** -- but was it generated from truly unseen data? Check that test data was held out before client partitioning.
- [ ] **"Model saved successfully"** -- but can it be loaded on a fresh Python session and produce the same predictions? Check serialization round-trip.
- [ ] **"Per-round metrics improve"** -- but are you averaging client metrics or evaluating the actual aggregated model? These are different things.
- [ ] **"Code runs on my machine"** -- but are file paths, data locations, and Python/Flower versions pinned and documented?

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1: Inf/NaN values in CICIDS2017 | Phase 1: Data Preprocessing | Assert zero Inf/NaN after cleaning; print value range per feature |
| P2: Class imbalance inflates accuracy | Phase 1: Data Preprocessing + Phase 3: Evaluation | Check F1/recall for attack class; confusion matrix shows balanced predictions |
| P3: Data leakage in train/test split | Phase 1: Data Preprocessing | Confirm identifier columns removed; scaler fitted on train only |
| P4: Parameter shape mismatch | Phase 2: FL Infrastructure | Single model.py imported everywhere; strict=True in load_state_dict |
| P5: Evaluating on training data | Phase 1: Data Partitioning + Phase 3: Evaluation | Separate global test set exists; client evaluate() uses local test split |
| P6: Column name inconsistencies | Phase 1: Data Preprocessing | Assert column count matches expectation; strip whitespace on load |
| P7: Non-convergence from learning rate | Phase 2-3: FL Training | Global loss trends downward over rounds; start conservative (E=1, lr=0.001) |
| P8: Missing DDoS files | Phase 1: Data Acquisition | Label value_counts shows both classes; DDoS count > 0 |
| P9: Flower API version issues | Phase 1: Project Setup | Pin flwr version; test basic client-server handshake before building pipeline |
| P10: Inconsistent feature scaling | Phase 1: Data Preprocessing | Global scaler computed before partitioning; verify feature ranges match across clients |

---

## Phase-Specific Warning Summary

### Phase 1 (Data + Setup) -- Highest Pitfall Density
This phase carries the most risk. Seven of ten critical pitfalls originate here. The preprocessing pipeline MUST be validated thoroughly before moving to FL training. Recommended validation gates:
- Zero Inf/NaN assertion
- Column name consistency assertion
- Both classes present in expected proportions
- Identifier columns removed
- Global scaler computed and saved
- Client partitions each contain both classes
- Train/test splits are clean (no leakage)

### Phase 2 (FL Infrastructure) -- Architecture Pitfalls
Flower setup pitfalls are less numerous but harder to debug. A parameter mismatch or API incompatibility wastes hours of debugging because the symptoms (bad metrics) look like data or model problems.

### Phase 3 (Training + Evaluation) -- Measurement Pitfalls
The most dangerous phase for false confidence. Everything "works" but metrics are misleading. The evaluation pipeline must be designed defensively: held-out data, proper metrics, confusion matrix, and per-client vs global metric distinction.

---

## Sources

- CICIDS2017 dataset documentation (University of New Brunswick CIC): known issues with Inf values, column inconsistencies, and class distribution are documented in the dataset description and extensively in follow-up academic analyses (Sharafaldin et al., 2018; Engelen et al., 2021).
- Flower framework documentation and tutorials (flower.ai): client/server API patterns, parameter handling, FedAvg implementation.
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017): original FedAvg paper describing client drift with local epoch/learning rate interaction.
- Li et al., "Federated Learning: Challenges, Methods, and Future Directions" (2020): survey covering convergence issues, data heterogeneity, and evaluation methodology.
- Multiple academic papers on FL-based IDS (2020-2024) commonly report class imbalance issues, preprocessing failures, and evaluation methodology concerns with CICIDS2017.
- Community issue discussions on Flower GitHub regarding API changes, NumPyClient deprecation patterns, and parameter serialization.

**Confidence note:** CICIDS2017-specific pitfalls (P1, P2, P3, P6, P8) are HIGH confidence -- these are among the most-reported issues in IDS literature and are dataset-inherent. FL training pitfalls (P4, P5, P7, P10) are HIGH confidence -- well-established in FL research. Flower API specifics (P9) are MEDIUM confidence -- the framework evolves; verify pinned version docs during implementation.

---
*Pitfalls research for: Federated Learning IDS for 6G Edge Networks*
*Researched: 2026-03-09*
