"""Entry point for running the data pipeline as a module.

Allows running the complete pipeline via:
    python -m federated_ids.data.preprocess

Or via the subpackage directly:
    python -m federated_ids.data
"""

from federated_ids.data.preprocess import main

main()
