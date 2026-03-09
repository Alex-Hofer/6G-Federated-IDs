"""Feature selection, normalization, and preprocessing for CICIDS2017.

Implements Stage 2 of the three-stage data pipeline (load -> preprocess -> partition).
Performs hybrid feature selection (domain shortlist + statistical filters),
StandardScaler normalization with no data leakage, global train/test split,
and class weight computation.

The preprocessing pipeline:
1. Drop identifier columns (IPs, ports, timestamps)
2. Apply domain-informed feature shortlist (DDoS-relevant flow features)
3. Drop zero-variance and near-constant features
4. Drop one from each highly correlated pair (|r| > threshold)
5. Split train/test globally (stratified, before any client partitioning)
6. Fit StandardScaler on training data ONLY, transform both splits
7. Compute class weights for balanced loss
8. Save artifacts (scaler, feature list, class weights) to processed directory

Example:
    >>> from federated_ids.data.loader import load_cicids2017
    >>> from federated_ids.data.preprocess import preprocess
    >>> df = load_cicids2017("data/raw", ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"])
    >>> config = {"data": {"test_size": 0.2, ...}, "seed": 42}
    >>> result = preprocess(df, config)
    >>> print(result["X_train"].shape, result["X_train"].dtype)
    (180000, 25) float32
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from federated_ids.data.loader import IDENTIFIER_COLS

logger = logging.getLogger(__name__)

# Domain-informed shortlist of DDoS-relevant features from CICIDS2017.
# These features are most informative for detecting volumetric and
# protocol-based DDoS attacks based on published IDS literature.
_DOMAIN_SHORTLIST = [
    # Flow-level features
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    # Inter-arrival time features
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    # Packet size features
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Average Packet Size",
    # Flag features (important for DDoS patterns)
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Down/Up Ratio",
    # Rate features
    "Fwd Packets/s",
    "Bwd Packets/s",
    # Header features
    "Fwd Header Length",
    "Bwd Header Length",
]


def select_features(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, dict]:
    """Select informative features using domain shortlist and statistical filters.

    Applies a hybrid feature selection strategy:
    1. Remove identifier columns (IPs, ports, timestamps)
    2. Start with domain-informed shortlist of DDoS-relevant features
    3. Fall back to all numeric columns if fewer than target remain
    4. Drop zero-variance features (variance < threshold)
    5. Drop near-constant features (>99% same value)
    6. Drop one from each highly correlated pair (|r| > threshold)

    Args:
        df: Cleaned DataFrame from ``load_cicids2017`` with binary Label column.
        config: Configuration dict with ``data.variance_threshold``,
            ``data.correlation_threshold``, and ``data.target_features``.

    Returns:
        A tuple of:
        - Filtered DataFrame with selected features and Label column
        - Feature drop report dict mapping dropped feature names to
          ``{"reason": str}`` entries
    """
    data_config = config["data"]
    variance_threshold = data_config.get("variance_threshold", 1e-10)
    correlation_threshold = data_config.get("correlation_threshold", 0.95)
    target_features = data_config.get("target_features", 30)

    feature_report: dict[str, dict[str, str]] = {}

    # Step 1: Drop identifier columns
    cols_to_drop = [col for col in IDENTIFIER_COLS if col in df.columns]
    for col in cols_to_drop:
        feature_report[col] = {"reason": "identifier column (data leakage risk)"}
    df = df.drop(columns=cols_to_drop, errors="ignore")
    logger.info("Dropped %d identifier columns", len(cols_to_drop))

    # Separate features from Label
    label_col = df["Label"].copy()
    feature_df = df.drop(columns=["Label"])

    # Keep only numeric columns
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in feature_df.columns if c not in numeric_cols]
    for col in non_numeric:
        feature_report[col] = {"reason": "non-numeric column"}
    feature_df = feature_df[numeric_cols]
    logger.info(
        "Dropped %d non-numeric columns. %d numeric features remain.",
        len(non_numeric),
        len(numeric_cols),
    )

    # Step 2: Domain shortlist -- keep features that exist in the data
    domain_cols = [col for col in _DOMAIN_SHORTLIST if col in feature_df.columns]
    if len(domain_cols) >= target_features:
        # Use domain shortlist
        dropped_by_domain = [
            c for c in feature_df.columns if c not in domain_cols
        ]
        for col in dropped_by_domain:
            feature_report[col] = {"reason": "not in domain shortlist"}
        feature_df = feature_df[domain_cols]
        logger.info(
            "Domain shortlist: kept %d features, dropped %d",
            len(domain_cols),
            len(dropped_by_domain),
        )
    else:
        # Fallback: keep all numeric columns
        logger.info(
            "Domain shortlist yielded only %d features (target: %d). "
            "Keeping all %d numeric columns.",
            len(domain_cols),
            target_features,
            len(feature_df.columns),
        )

    # Step 3: Drop zero-variance features
    variances = feature_df.var()
    zero_var_cols = variances[variances < variance_threshold].index.tolist()
    for col in zero_var_cols:
        feature_report[col] = {
            "reason": f"zero variance (var={variances[col]:.2e} < {variance_threshold:.2e})"
        }
    feature_df = feature_df.drop(columns=zero_var_cols)
    logger.info("Dropped %d zero-variance features", len(zero_var_cols))

    # Step 4: Drop near-constant features (>99% same value)
    near_constant_cols = []
    for col in feature_df.columns:
        most_common_pct = feature_df[col].value_counts(normalize=True).iloc[0]
        if most_common_pct > 0.99:
            near_constant_cols.append(col)
            feature_report[col] = {
                "reason": f"near-constant ({most_common_pct:.1%} same value)"
            }
    feature_df = feature_df.drop(columns=near_constant_cols)
    logger.info("Dropped %d near-constant features", len(near_constant_cols))

    # Step 5: Drop highly correlated features
    if len(feature_df.columns) > 1:
        corr_matrix = feature_df.corr().abs()
        # Upper triangle mask (exclude diagonal)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )

        corr_drop = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > correlation_threshold].tolist()
            for corr_col in high_corr:
                if corr_col not in corr_drop:
                    # Drop the feature with lower variance
                    var_col = variances.get(col, 0)
                    var_corr = variances.get(corr_col, 0)
                    drop_target = corr_col if var_corr <= var_col else col
                    if drop_target not in corr_drop:
                        corr_drop.add(drop_target)
                        partner = col if drop_target == corr_col else corr_col
                        r_val = corr_matrix.loc[col, corr_col]
                        feature_report[drop_target] = {
                            "reason": (
                                f"high correlation with '{partner}' "
                                f"(|r|={r_val:.3f} > {correlation_threshold})"
                            )
                        }

        feature_df = feature_df.drop(columns=list(corr_drop))
        logger.info("Dropped %d highly correlated features", len(corr_drop))

    # Log final feature report
    logger.info("--- Feature Selection Report ---")
    logger.info("Features dropped: %d", len(feature_report))
    for feat, info in feature_report.items():
        logger.info("  DROPPED: %-35s  Reason: %s", feat, info["reason"])
    logger.info("Features selected: %d", len(feature_df.columns))
    for feat in sorted(feature_df.columns):
        logger.info("  KEPT:    %s", feat)
    logger.info("--- End Feature Selection Report ---")

    # Reattach Label column
    result_df = feature_df.copy()
    result_df["Label"] = label_col.values

    return result_df, feature_report


def preprocess(df: pd.DataFrame, config: dict) -> dict:
    """Run the full preprocessing pipeline on a cleaned CICIDS2017 DataFrame.

    Performs feature selection, train/test split, StandardScaler normalization
    (fitted on training data only to prevent data leakage), class weight
    computation, and saves all artifacts to the processed directory.

    Args:
        df: Cleaned DataFrame from ``load_cicids2017`` with binary Label column
            and stripped column names.
        config: Configuration dict with keys:
            - ``data.test_size``: Fraction for test split (default 0.2)
            - ``data.processed_dir``: Directory for saving artifacts
            - ``data.variance_threshold``: Threshold for zero-variance filter
            - ``data.correlation_threshold``: Threshold for correlation filter
            - ``data.target_features``: Target feature count (informational)
            - ``seed``: Random seed for reproducibility

    Returns:
        Dict with keys:
        - ``X_train``: Training features as float32 ndarray
        - ``X_test``: Test features as float32 ndarray
        - ``y_train``: Training labels as int64 ndarray
        - ``y_test``: Test labels as int64 ndarray
        - ``scaler``: Fitted StandardScaler instance
        - ``feature_names``: List of selected feature names
        - ``class_weights``: Dict mapping class labels to weights
        - ``feature_report``: Dict of dropped features and reasons
    """
    data_config = config["data"]
    test_size = data_config.get("test_size", 0.2)
    processed_dir = data_config.get("processed_dir", "./data/processed")
    seed = config.get("seed", 42)

    # Step 1: Feature selection
    filtered_df, feature_report = select_features(df, config)

    # Step 2: Extract X (features) and y (labels)
    feature_names = [c for c in filtered_df.columns if c != "Label"]
    X = filtered_df[feature_names].values.astype(np.float32)
    y = filtered_df["Label"].values.astype(np.int64)

    logger.info(
        "Feature matrix shape: %s, Label shape: %s", X.shape, y.shape
    )

    # Step 3: Global train/test split (BEFORE any client partitioning)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    logger.info(
        "Train/test split: train=%d, test=%d (test_size=%.2f)",
        len(y_train),
        len(y_test),
        test_size,
    )

    # Step 4: Fit StandardScaler on training data ONLY (Pitfall 3)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    logger.info(
        "StandardScaler fitted on training data only. "
        "Train mean near 0: %s, Test mean NOT near 0: %s",
        np.allclose(X_train.mean(axis=0), 0, atol=1e-6),
        not np.allclose(X_test.mean(axis=0), 0, atol=0.05),
    )

    # Step 5: Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(cls): float(w) for cls, w in zip(classes, weights)}

    logger.info("Class weights: %s", class_weights)

    # Log class distribution per split
    for split_name, y_split in [("train", y_train), ("test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        total = len(y_split)
        dist = {int(u): f"{c} ({c / total * 100:.1f}%)" for u, c in zip(unique, counts)}
        logger.info("  %s class distribution: %s", split_name, dist)

    # Step 6: Save artifacts to processed directory
    os.makedirs(processed_dir, exist_ok=True)

    # Save fitted scaler
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info("Saved scaler to %s", scaler_path)

    # Save selected feature names
    features_path = os.path.join(processed_dir, "features.json")
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("Saved %d feature names to %s", len(feature_names), features_path)

    # Save class weights
    weights_path = os.path.join(processed_dir, "class_weights.json")
    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=2)
    logger.info("Saved class weights to %s", weights_path)

    # Save class distribution stats
    stats = {}
    for split_name, y_split in [("train", y_train), ("test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        stats[split_name] = {
            int(u): int(c) for u, c in zip(unique, counts)
        }
    stats_path = os.path.join(processed_dir, "class_distribution.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved class distribution to %s", stats_path)

    # --- VALIDATION GATES ---
    assert not np.isnan(X_train).any(), (
        "VALIDATION FAILED: NaN values in X_train after scaling"
    )
    assert not np.isnan(X_test).any(), (
        "VALIDATION FAILED: NaN values in X_test after scaling"
    )
    assert not np.isinf(X_train).any(), (
        "VALIDATION FAILED: Inf values in X_train after scaling"
    )
    assert not np.isinf(X_test).any(), (
        "VALIDATION FAILED: Inf values in X_test after scaling"
    )
    assert X_train.dtype == np.float32, (
        f"VALIDATION FAILED: X_train dtype is {X_train.dtype}, expected float32"
    )
    assert X_test.dtype == np.float32, (
        f"VALIDATION FAILED: X_test dtype is {X_test.dtype}, expected float32"
    )

    logger.info(
        "Preprocessing complete. Features: %d, Train: %d, Test: %d",
        len(feature_names),
        len(y_train),
        len(y_test),
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "class_weights": class_weights,
        "feature_report": feature_report,
    }
