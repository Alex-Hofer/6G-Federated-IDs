"""End-to-end integration test for the full federated IDS pipeline.

Exercises the complete pipeline (preprocess -> FL training -> evaluation)
using synthetic CICIDS2017 data, validating that all output artifacts
are produced without requiring the real dataset.
"""

import json
import os

import numpy as np
import pytest
import yaml

from tests.conftest import _CICIDS_COLUMNS


@pytest.fixture
def integration_env(tmp_path):
    """Set up a self-contained pipeline environment with synthetic CSV data.

    Creates a synthetic CICIDS2017 CSV (~500 rows) with realistic column
    structure, injected inf/nan values, and a minimal config YAML tuned
    for fast execution (~2 clients, 2 rounds, 1 local epoch).

    Returns:
        Tuple of ``(config_path, config_dict)``.
    """
    rng = np.random.RandomState(42)
    n_rows = 500

    # Generate synthetic feature data
    data = {}
    for col in _CICIDS_COLUMNS:
        data[col] = rng.rand(n_rows).astype(np.float64) * 1000

    # Inject np.inf into Flow Bytes/s (row 5) and Flow Packets/s (row 10)
    data["Flow Bytes/s"][5] = np.inf
    data[" Flow Packets/s"][10] = np.inf

    # Inject np.nan into Flow Duration (row 15)
    data[" Flow Duration"][15] = np.nan

    # Label column: 300 BENIGN + 200 DDoS, shuffled
    labels = ["BENIGN"] * 300 + ["DDoS"] * 200
    rng.shuffle(labels)
    data[" Label"] = labels

    # Write CSV
    import pandas as pd

    df = pd.DataFrame(data)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / "synthetic.csv"
    df.to_csv(csv_path, index=False)

    # Write config YAML
    config = {
        "data": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "files": ["synthetic.csv"],
            "test_size": 0.2,
            "target_features": 30,
            "correlation_threshold": 0.95,
            "variance_threshold": 1.0e-10,
        },
        "model": {
            "hidden_layers": [16, 8],
            "dropout": 0.1,
            "num_classes": 2,
        },
        "training": {
            "learning_rate": 0.001,
            "local_epochs": 1,
            "batch_size": 32,
            "weighted_loss": True,
        },
        "federation": {
            "num_clients": 2,
            "num_rounds": 2,
            "fraction_fit": 1.0,
        },
        "seed": 42,
        "output_dir": str(tmp_path / "outputs"),
        "log_level": "WARNING",
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path), config


@pytest.mark.slow
class TestFullPipeline:
    """Integration tests running the full pipeline on synthetic data."""

    def test_pipeline_produces_all_outputs(self, integration_env):
        """Run full pipeline and verify all expected output artifacts exist."""
        config_path, config = integration_env
        output_dir = config["output_dir"]

        from federated_ids.pipeline import run_pipeline

        run_pipeline(config_path)

        # Assert checkpoint exists (MODL-03 validation)
        checkpoint_path = os.path.join(output_dir, "checkpoints", "global_model.pt")
        assert os.path.isfile(checkpoint_path), (
            f"Global model checkpoint not found: {checkpoint_path}"
        )

        # Assert metrics JSON is valid (INFR-02 -- config snapshot for reproducibility)
        metrics_path = os.path.join(output_dir, "metrics", "fl_metrics.json")
        assert os.path.isfile(metrics_path), (
            f"FL metrics JSON not found: {metrics_path}"
        )
        with open(metrics_path) as f:
            metrics_data = json.load(f)
        assert "rounds" in metrics_data, "FL metrics missing 'rounds' key"
        assert "config" in metrics_data, "FL metrics missing 'config' key"

        # Assert plots exist and are non-empty
        plots_dir = os.path.join(output_dir, "plots")
        for plot_name in ["confusion_matrix.png", "convergence.png", "client_comparison.png"]:
            plot_path = os.path.join(plots_dir, plot_name)
            assert os.path.isfile(plot_path), f"Plot not found: {plot_path}"
            assert os.path.getsize(plot_path) > 0, f"Plot is empty: {plot_path}"

        # Assert classification report exists
        report_path = os.path.join(plots_dir, "classification_report.txt")
        assert os.path.isfile(report_path), (
            f"Classification report not found: {report_path}"
        )
