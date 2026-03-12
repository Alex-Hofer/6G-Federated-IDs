"""Entry point for running standalone model training as a module.

Allows running via:
    python -m federated_ids.model.train
"""

from federated_ids.model.train import standalone_train

standalone_train()
