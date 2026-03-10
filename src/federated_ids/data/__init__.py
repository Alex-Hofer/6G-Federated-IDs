"""Data loading, preprocessing, and partitioning for CICIDS2017.

This subpackage handles the complete data pipeline:
- Loading raw CICIDS2017 CSV files
- Cleaning, feature selection, and normalization
- IID partitioning across federated clients
- PyTorch DataLoader creation
"""

from federated_ids.data.loader import load_cicids2017
from federated_ids.data.partition import create_dataloaders, partition_iid
from federated_ids.data.preprocess import main as run_pipeline
from federated_ids.data.preprocess import preprocess, select_features

__all__ = [
    "load_cicids2017",
    "preprocess",
    "select_features",
    "run_pipeline",
    "partition_iid",
    "create_dataloaders",
]
