"""End-to-end pipeline orchestrator for federated IDS experiments.

Chains all stages (preprocess, FL training, evaluation) into a single
command, providing a zero-config way to run the complete experiment.

Usage::

    federated-ids-run-all --config config/default.yaml
    python -m federated_ids.pipeline --config config/default.yaml

Example:
    >>> from federated_ids.pipeline import run_pipeline
    >>> run_pipeline("config/default.yaml")
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from federated_ids.config import load_config
from federated_ids.seed import set_global_seed

logger = logging.getLogger(__name__)


def _format_size(size_bytes: int) -> str:
    """Format a file size in human-readable units.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Human-readable string (e.g. ``"1.2 MB"``).
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _print_pipeline_summary(output_dir: str) -> None:
    """Print an end-of-run summary listing output files and key metrics.

    Walks the output directory, lists all produced files with sizes,
    and extracts final-round metrics from the FL metrics JSON.

    Args:
        output_dir: Path to the pipeline output directory.
    """
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)

    # List all output files
    logger.info("Output files:")
    for dirpath, _dirnames, filenames in os.walk(output_dir):
        for fname in sorted(filenames):
            fpath = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(fpath, output_dir)
            size = os.path.getsize(fpath)
            logger.info("  %-45s %s", rel_path, _format_size(size))

    # Extract final-round metrics from fl_metrics.json
    metrics_path = os.path.join(output_dir, "metrics", "fl_metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics_data = json.load(f)
        rounds = metrics_data.get("rounds", [])
        if rounds:
            final = rounds[-1]
            logger.info("-" * 60)
            logger.info("Final round metrics (round %d):", final.get("round", len(rounds)))
            logger.info("  F1:        %.4f", final.get("f1", 0.0))
            logger.info("  Precision: %.4f", final.get("precision", 0.0))
            logger.info("  Recall:    %.4f", final.get("recall", 0.0))
            logger.info("  Accuracy:  %.4f", final.get("accuracy", 0.0))
            logger.info("  Loss:      %.4f", final.get("loss", 0.0))

    logger.info("-" * 60)
    logger.info("Pipeline complete.")


def run_pipeline(config_path: str = "config/default.yaml") -> None:
    """Run the full federated IDS pipeline end-to-end.

    Chains preprocessing, federated training, and evaluation in
    sequence.  Skips preprocessing when cached tensors exist.

    Args:
        config_path: Path to the YAML configuration file.

    Raises:
        Any exception from the underlying stages propagates immediately
        (fail-fast behaviour).
    """
    # Load config and set reproducibility seed
    config = load_config(config_path)
    set_global_seed(config.get("seed", 42))

    # Configure logging once at the top level
    log_level = config.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    output_dir = config.get("output_dir", "./outputs")
    processed_dir = config["data"].get("processed_dir", "./data/processed")

    # ------------------------------------------------------------------
    # Stage 1/3: Preprocessing
    # ------------------------------------------------------------------
    logger.info("=== Stage 1/3: Preprocessing ===")

    from federated_ids.data.preprocess import _cache_exists

    if _cache_exists(processed_dir):
        logger.info("Cached tensors found in %s -- skipping preprocessing.", processed_dir)
    else:
        from federated_ids.data.preprocess import main as preprocess_main

        preprocess_main(config_path)

    # ------------------------------------------------------------------
    # Stage 2/3: Federated Training
    # ------------------------------------------------------------------
    logger.info("=== Stage 2/3: Federated Training ===")

    from federated_ids.fl.server import run_federated_training

    run_federated_training(config, config_path=config_path)

    # ------------------------------------------------------------------
    # Stage 3/3: Evaluation
    # ------------------------------------------------------------------
    logger.info("=== Stage 3/3: Evaluation ===")

    from federated_ids.eval.__main__ import main as eval_main

    eval_main(config_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_pipeline_summary(output_dir)


def main() -> None:
    """CLI entry point for ``federated-ids-run-all``."""
    parser = argparse.ArgumentParser(
        description="Run the full federated IDS pipeline (preprocess -> train -> evaluate)."
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
