#!/usr/bin/env python
"""Phase 4 requirement verification script.

Checks EVAL-02, EVAL-03, EVAL-04 using synthetic data.
Designed to be run standalone: ``python scripts/verify_phase4.py``.

Exit codes:
    0 -- all checks passed
    1 -- one or more checks failed
"""

import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless rendering -- before any pyplot import

import numpy as np  # noqa: E402
import torch  # noqa: E402


# ---------------------------------------------------------------------------
# EVAL-02: Confusion matrix and classification report
# ---------------------------------------------------------------------------

def check_eval_02(tmp_dir: str) -> dict:
    """Verify confusion matrix PNG and classification report text output."""
    from federated_ids.eval.plots import plot_confusion_matrix, save_classification_report

    # Generate synthetic predictions: 100 samples, ~70% class 0, ~30% class 1
    rng = np.random.RandomState(42)
    y_true = np.concatenate([np.zeros(70, dtype=int), np.ones(30, dtype=int)])
    # Introduce some errors: flip ~10% of predictions
    y_pred = y_true.copy()
    flip_idx = rng.choice(100, size=10, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]

    eval_dir = os.path.join(tmp_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    cr_path = os.path.join(eval_dir, "classification_report.txt")

    try:
        plot_confusion_matrix(y_true, y_pred, cm_path)
        save_classification_report(y_true, y_pred, cr_path)

        # Verify confusion matrix PNG
        cm_exists = os.path.isfile(cm_path)
        cm_size = os.path.getsize(cm_path) if cm_exists else 0

        cm_valid = False
        if cm_exists and cm_size > 0:
            from PIL import Image
            img = Image.open(cm_path)
            img.verify()
            cm_valid = True

        # Verify classification report
        cr_exists = os.path.isfile(cr_path)
        cr_content = ""
        if cr_exists:
            with open(cr_path) as f:
                cr_content = f.read()

        cr_has_benign = "BENIGN" in cr_content
        cr_has_ddos = "DDoS" in cr_content
        cr_has_precision = "precision" in cr_content
        cr_has_recall = "recall" in cr_content
        cr_valid = cr_has_benign and cr_has_ddos and cr_has_precision and cr_has_recall

        all_ok = cm_valid and cr_valid
        status = "PASS" if all_ok else "FAIL"
        value = (
            f"cm_png={cm_size}B valid={cm_valid}, "
            f"report: BENIGN={cr_has_benign} DDoS={cr_has_ddos} "
            f"precision={cr_has_precision} recall={cr_has_recall}"
        )

    except Exception as exc:
        status = "FAIL"
        value = f"Exception: {exc}"

    return {
        "req_id": "EVAL-02",
        "check": "Confusion matrix PNG + classification report text",
        "status": status,
        "value": value,
        "method": "plot_confusion_matrix() + save_classification_report() with synthetic data",
    }


# ---------------------------------------------------------------------------
# EVAL-03: Convergence plots
# ---------------------------------------------------------------------------

def check_eval_03(tmp_dir: str) -> dict:
    """Verify convergence and client comparison plot generation."""
    from federated_ids.eval.plots import plot_client_comparison, plot_convergence

    eval_dir = os.path.join(tmp_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    try:
        # Create synthetic fl_metrics.json with realistic convergence curves
        rng = np.random.RandomState(42)
        rounds_data = []
        for i in range(1, 6):
            # Exponential decay for loss (0.8 -> 0.3), growth for F1 (0.5 -> 0.85)
            t = (i - 1) / 4.0  # normalize to [0, 1]
            loss = 0.8 * np.exp(-1.5 * t) + rng.normal(0, 0.02)
            f1 = 0.5 + 0.35 * (1 - np.exp(-2.0 * t)) + rng.normal(0, 0.02)
            rounds_data.append({
                "round": i,
                "loss": float(max(0.05, loss)),
                "f1": float(min(0.99, max(0.1, f1))),
                "accuracy": float(min(0.99, max(0.1, f1 + 0.05))),
                "precision": float(min(0.99, max(0.1, f1 + 0.03))),
                "recall": float(min(0.99, max(0.1, f1 - 0.02))),
            })

        metrics_path = os.path.join(tmp_dir, "fl_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"rounds": rounds_data, "config": {}}, f, indent=2)

        # Generate convergence plot
        conv_path = os.path.join(eval_dir, "convergence.png")
        plot_convergence(metrics_path, conv_path)

        # Generate client comparison plot
        client_metrics = [
            {"f1": 0.80, "precision": 0.85, "recall": 0.75},
            {"f1": 0.78, "precision": 0.82, "recall": 0.74},
            {"f1": 0.83, "precision": 0.88, "recall": 0.78},
        ]
        global_metrics = {"f1": 0.85, "precision": 0.90, "recall": 0.80}
        comp_path = os.path.join(eval_dir, "client_comparison.png")
        plot_client_comparison(client_metrics, global_metrics, comp_path)

        # Verify convergence PNG
        conv_exists = os.path.isfile(conv_path)
        conv_size = os.path.getsize(conv_path) if conv_exists else 0
        conv_valid = False
        conv_dims = ""
        if conv_exists and conv_size > 0:
            from PIL import Image
            img = Image.open(conv_path)
            conv_dims = f"{img.size[0]}x{img.size[1]}"
            img.verify()
            conv_valid = True

        # Verify client comparison PNG
        comp_exists = os.path.isfile(comp_path)
        comp_size = os.path.getsize(comp_path) if comp_exists else 0
        comp_valid = False
        comp_dims = ""
        if comp_exists and comp_size > 0:
            from PIL import Image
            img = Image.open(comp_path)
            comp_dims = f"{img.size[0]}x{img.size[1]}"
            img.verify()
            comp_valid = True

        all_ok = conv_valid and comp_valid
        status = "PASS" if all_ok else "FAIL"
        value = (
            f"convergence={conv_size}B {conv_dims}, "
            f"comparison={comp_size}B {comp_dims}"
        )

    except Exception as exc:
        status = "FAIL"
        value = f"Exception: {exc}"

    return {
        "req_id": "EVAL-03",
        "check": "Convergence + client comparison plot generation",
        "status": status,
        "value": value,
        "method": "plot_convergence() + plot_client_comparison() with synthetic metrics",
    }


# ---------------------------------------------------------------------------
# EVAL-04: TensorBoard logging
# ---------------------------------------------------------------------------

def check_eval_04(tmp_dir: str) -> dict:
    """Verify TensorBoard scalar logging during federated training."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {
            "req_id": "EVAL-04",
            "check": "TensorBoard scalar logging during FL training",
            "status": "FAIL",
            "value": "tensorboard not installed",
            "method": "import check failed",
        }

    try:
        from federated_ids.fl.server import run_federated_training

        # Create synthetic cached tensors
        processed_dir = os.path.join(tmp_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        rng = np.random.RandomState(42)
        n_samples, n_features = 100, 10

        X = torch.tensor(rng.randn(n_samples, n_features), dtype=torch.float32)
        y = torch.tensor((rng.rand(n_samples) < 0.3).astype(np.int64))

        torch.save(X, os.path.join(processed_dir, "X_train.pt"))
        torch.save(y, os.path.join(processed_dir, "y_train.pt"))
        torch.save(X[:20], os.path.join(processed_dir, "X_test.pt"))
        torch.save(y[:20], os.path.join(processed_dir, "y_test.pt"))

        # class_weights.json (needed if weighted_loss is true, but we set false)
        weights = {"0": 0.6, "1": 1.8}
        with open(os.path.join(processed_dir, "class_weights.json"), "w") as f:
            json.dump(weights, f)

        output_dir = os.path.join(tmp_dir, "outputs")
        tb_dir = os.path.join(output_dir, "tensorboard")

        config = {
            "data": {
                "raw_dir": os.path.join(tmp_dir, "raw"),
                "processed_dir": processed_dir,
                "files": [],
                "test_size": 0.2,
                "target_features": 10,
                "correlation_threshold": 0.95,
                "variance_threshold": 1e-10,
            },
            "model": {
                "input_dim": n_features,
                "hidden_layers": [16, 8],
                "num_classes": 2,
                "dropout": 0.1,
            },
            "training": {
                "learning_rate": 0.001,
                "local_epochs": 1,
                "batch_size": 32,
                "weighted_loss": False,
            },
            "federation": {
                "num_clients": 2,
                "num_rounds": 2,
                "fraction_fit": 1.0,
            },
            "seed": 42,
            "output_dir": output_dir,
            "log_level": "WARNING",
        }

        # Run actual federated training (writes real TB events)
        run_federated_training(config)

        # Read TensorBoard events
        ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
        ea.Reload()

        scalar_tags = ea.Tags().get("scalars", [])

        expected_tags = [
            "Global/loss",
            "Global/accuracy",
            "Global/f1",
            "Global/precision",
            "Global/recall",
        ]

        found_tags = [t for t in expected_tags if t in scalar_tags]
        missing_tags = [t for t in expected_tags if t not in scalar_tags]

        # Check event counts (should have 2 events per tag for 2 rounds)
        event_counts = {}
        for tag in found_tags:
            events = ea.Scalars(tag)
            event_counts[tag] = len(events)

        all_tags_found = len(missing_tags) == 0
        all_counts_correct = all(c == 2 for c in event_counts.values())

        all_ok = all_tags_found and all_counts_correct
        status = "PASS" if all_ok else "FAIL"

        tags_str = ", ".join(f"{t}={event_counts.get(t, 0)}" for t in expected_tags)
        value = f"tags: {tags_str}"
        if missing_tags:
            value += f" | missing: {missing_tags}"

    except Exception as exc:
        status = "FAIL"
        value = f"Exception: {exc}"

    return {
        "req_id": "EVAL-04",
        "check": "TensorBoard scalar logging during FL training",
        "status": status,
        "value": value,
        "method": "run_federated_training() + EventAccumulator tag/event verification",
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: list[dict]) -> None:
    """Print a formatted verification results table."""
    header = (
        f"| {'REQ ID':<10} "
        f"| {'Check':<52} "
        f"| {'Method':<55} "
        f"| {'Result':<6} "
        f"| {'Value'} |"
    )
    separator = (
        f"|{'-' * 12}"
        f"|{'-' * 54}"
        f"|{'-' * 57}"
        f"|{'-' * 8}"
        f"|{'-' * 60}|"
    )

    print()
    print("=" * 120)
    print("PHASE 4 REQUIREMENT VERIFICATION RESULTS")
    print("=" * 120)
    print(header)
    print(separator)

    for r in results:
        print(
            f"| {r['req_id']:<10} "
            f"| {r['check']:<52} "
            f"| {r['method']:<55} "
            f"| {r['status']:<6} "
            f"| {r['value'][:58]:<58} |"
        )

    print(separator)

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all Phase 4 requirement checks and print results."""
    # Environment header
    print("=" * 80)
    print("Phase 4 Requirement Verification")
    print("=" * 80)
    print(f"Date:     {datetime.now().isoformat()}")
    print(f"Python:   {sys.version}")
    print(f"OS:       {platform.platform()}")
    print(f"CWD:      {os.getcwd()}")
    print("=" * 80)
    print()

    # Suppress verbose logging from training / FL loops
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Create temp directory for all outputs
    tmp_dir = tempfile.mkdtemp(prefix="verify_phase4_")
    results: list[dict] = []

    try:
        # Check EVAL-02: Confusion matrix + classification report
        print("[1/3] Checking EVAL-02: Confusion matrix and classification report...")
        r1 = check_eval_02(tmp_dir)
        results.append(r1)
        print(f"       {r1['status']}: {r1['value']}")
        print()

        # Check EVAL-03: Convergence plots
        print("[2/3] Checking EVAL-03: Convergence and client comparison plots...")
        r2 = check_eval_03(tmp_dir)
        results.append(r2)
        print(f"       {r2['status']}: {r2['value']}")
        print()

        # Check EVAL-04: TensorBoard logging
        print("[3/3] Checking EVAL-04: TensorBoard scalar logging...")
        r3 = check_eval_04(tmp_dir)
        results.append(r3)
        print(f"       {r3['status']}: {r3['value']}")
        print()

        # Print summary table
        print_summary_table(results)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Exit code
    all_passed = all(r["status"] == "PASS" for r in results)
    final = "PASS" if all_passed else "FAIL"
    print(f"Phase 4 Verification: {final}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
