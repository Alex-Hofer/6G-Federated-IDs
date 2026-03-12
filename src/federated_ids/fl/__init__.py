"""Federated learning infrastructure with pure-Python FedAvg.

This subpackage implements the FL client and server logic
for distributed model training across simulated edge nodes.
Uses plain NumPy parameter transport (no Flower dependency)
while maintaining API compatibility for future Flower migration.
"""

from federated_ids.fl.client import FederatedClient
from federated_ids.fl.server import (
    check_convergence,
    fedavg_aggregate,
    print_config_banner,
    run_federated_training,
    save_fl_metrics,
    server_evaluate,
)

__all__ = [
    "FederatedClient",
    "check_convergence",
    "fedavg_aggregate",
    "print_config_banner",
    "run_federated_training",
    "save_fl_metrics",
    "server_evaluate",
]
