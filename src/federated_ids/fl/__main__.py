"""Entry point for federated training: ``python -m federated_ids.fl``.

Provides a CLI for running the complete federated training pipeline
with configurable number of clients and rounds via command-line overrides.

Usage::

    python -m federated_ids.fl --config config/default.yaml --num-clients 5 --num-rounds 10
    federated-ids-train-fl --num-rounds 30
"""

from __future__ import annotations

import argparse

from federated_ids.config import load_config
from federated_ids.fl.server import run_federated_training


def main(
    config_path: str | None = None,
    num_clients: int | None = None,
    num_rounds: int | None = None,
) -> None:
    """Run federated training with optional CLI overrides.

    Supports both CLI invocation (``federated-ids-train-fl``) and
    programmatic use from other modules.

    Args:
        config_path: Path to YAML config file. If ``None``, parsed from
            CLI ``--config`` argument (default ``config/default.yaml``).
        num_clients: Override for ``federation.num_clients``. If ``None``,
            parsed from CLI ``--num-clients`` argument.
        num_rounds: Override for ``federation.num_rounds``. If ``None``,
            parsed from CLI ``--num-rounds`` argument.
    """
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="Run federated training across simulated edge clients."
        )
        parser.add_argument(
            "--config",
            default="config/default.yaml",
            help="Path to YAML configuration file (default: config/default.yaml)",
        )
        parser.add_argument(
            "--num-clients",
            type=int,
            default=None,
            help="Override number of federated clients",
        )
        parser.add_argument(
            "--num-rounds",
            type=int,
            default=None,
            help="Override number of federated rounds",
        )
        args = parser.parse_args()
        config_path = args.config
        num_clients = args.num_clients
        num_rounds = args.num_rounds

    config = load_config(config_path)

    # Apply CLI overrides
    if num_clients is not None:
        config["federation"]["num_clients"] = num_clients
    if num_rounds is not None:
        config["federation"]["num_rounds"] = num_rounds

    run_federated_training(config, config_path=config_path)


if __name__ == "__main__":
    main()
