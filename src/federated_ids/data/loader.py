"""CICIDS2017 CSV data loading with cleaning and label mapping.

Loads raw CICIDS2017 MachineLearningCSV files, handles known data quality
issues (whitespace in column names, Inf/NaN values from CICFlowMeter bugs),
and maps multi-class labels to binary classification (BENIGN=0, DDoS=1).

This module implements Stage 1 of the three-stage data pipeline
(load -> preprocess -> partition).

Example:
    >>> from federated_ids.data.loader import load_cicids2017
    >>> df = load_cicids2017("data/raw", ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"])
    >>> print(df.shape)
    (225000, 79)
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DDoS attack label variants found in CICIDS2017 dataset.
# All are mapped to 1 (attack) during binary label conversion.
DDOS_LABELS = {
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
}

# Identifier columns that must be dropped before feature selection.
# These carry no generalizable signal and can cause data leakage
# (model memorizes IPs/ports instead of learning attack signatures).
IDENTIFIER_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
]


def load_cicids2017(data_dir: str, files: list[str]) -> pd.DataFrame:
    """Load and clean CICIDS2017 CSV files for binary DDoS classification.

    Performs the following cleaning steps in order:
    1. Load each CSV and strip whitespace from column names
    2. Concatenate all DataFrames
    3. Strip whitespace from Label values
    4. Replace Inf values with NaN
    5. Drop all rows containing NaN
    6. Map labels to binary: BENIGN -> 0, all DDoS variants -> 1
    7. Validate: no Inf, no NaN, labels are {0, 1}

    Args:
        data_dir: Path to the directory containing raw CSV files.
        files: List of CSV filenames to load from ``data_dir``.

    Returns:
        Cleaned DataFrame with stripped column names, no Inf/NaN values,
        and binary labels (0=benign, 1=DDoS) in the ``Label`` column.

    Raises:
        FileNotFoundError: If any specified CSV file does not exist.
        AssertionError: If validation gates fail (Inf/NaN remain or
            labels are not binary).
    """
    dfs = []

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"CSV file not found: {filepath}"
            )

        logger.info("Loading %s", filepath)
        df = pd.read_csv(filepath)

        # Strip whitespace from column names (Pitfall 2)
        df.columns = df.columns.str.strip()

        dfs.append(df)
        logger.info("  Loaded %d rows, %d columns", len(df), len(df.columns))

    combined = pd.concat(dfs, ignore_index=True)
    total_loaded = len(combined)
    logger.info("Total rows loaded: %d", total_loaded)

    # Strip whitespace from Label values (e.g., " BENIGN" -> "BENIGN")
    combined["Label"] = combined["Label"].str.strip()

    # Replace Inf with NaN, then drop all NaN rows (Pitfall 1)
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)

    rows_dropped = total_loaded - len(combined)
    logger.info(
        "Dropped %d rows (Inf/NaN). Remaining: %d", rows_dropped, len(combined)
    )

    # Map labels to binary: BENIGN -> 0, DDoS variants -> 1
    combined["Label"] = combined["Label"].apply(
        lambda x: 1 if x in DDOS_LABELS else 0
    )

    # Log class distribution
    class_counts = combined["Label"].value_counts().sort_index()
    total = len(combined)
    for label, count in class_counts.items():
        pct = count / total * 100
        logger.info("  Class %d: %d rows (%.1f%%)", label, count, pct)

    # --- VALIDATION GATES ---
    # These assertions catch silent data corruption early.
    numeric_df = combined.select_dtypes(include=[np.number])
    assert not np.isinf(
        numeric_df.values
    ).any(), "VALIDATION FAILED: Inf values remain after cleaning"

    assert (
        not combined.isna().any().any()
    ), "VALIDATION FAILED: NaN values remain after cleaning"

    unique_labels = set(combined["Label"].unique())
    assert unique_labels == {
        0,
        1,
    }, f"VALIDATION FAILED: Labels are {unique_labels}, expected {{0, 1}}"

    logger.info(
        "Validation passed: no Inf, no NaN, binary labels {0, 1}. "
        "Final shape: %s",
        combined.shape,
    )

    return combined
