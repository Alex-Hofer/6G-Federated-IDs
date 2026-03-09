"""Data loading, preprocessing, and partitioning for CICIDS2017.

This subpackage handles the complete data pipeline:
- Loading raw CICIDS2017 CSV files
- Cleaning, feature selection, and normalization
- IID partitioning across federated clients
- PyTorch DataLoader creation
"""
