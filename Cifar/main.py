from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Type, TypedDict

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from classifier import (
    LogisticRegressionClassifier,
    NeuralNetworkClassifier,
    MNISTClassifier,
    CIFARMLPClassifier,
    CIFARCNNClassifier,
    CIFARCNNImprovedClassifier,
)

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"


class ModelSpec(TypedDict):
    classifier: Type[MNISTClassifier]
    dataset: str


MODEL_TYPES: Dict[str, ModelSpec] = {
    "logistic": {"classifier": LogisticRegressionClassifier, "dataset": "mnist"},
    "nn": {"classifier": NeuralNetworkClassifier, "dataset": "mnist"},
    "cifar_mlp": {"classifier": CIFARMLPClassifier, "dataset": "cifar10"},
    "cnn": {"classifier": CIFARCNNClassifier, "dataset": "cifar10"},
    "cnn_improved": {"classifier": CIFARCNNImprovedClassifier, "dataset": "cifar10"},
}


def _run_name(model_type: str, tag: str | None = None, augment: bool = False) -> str:
    """Resolve the artifact name for a run.

    Artifacts are keyed on the *run name*, not the model type, so that variants
    of the same model (e.g. CNN with and without augmentation) do not overwrite
    each other. An explicit --tag always wins; otherwise --augment appends the
    "_aug" suffix and a plain run keeps the bare model type.
    """
    if tag:
        return tag
    return f"{model_type}_aug" if augment else model_type


def _model_path(run: str) -> str:
    return os.path.join(MODEL_DIR, f"{run}_model.keras")


def _history_path(run: str) -> str:
    return os.path.join(RESULTS_DIR, f"{run}_history.json")


def _data_path(dataset: str) -> str:
    return os.path.join(DATA_DIR, f"{dataset}.npz")


def _configure_mnist() -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Flatten 28x28 images to 784-dim vectors and normalise to [0, 1]
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    np.savez(
        _data_path("mnist"),
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"Dataset saved to {_data_path('mnist')}")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples:     {len(x_test)}")


def _configure_cifar10() -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Keep image tensors for CNN and normalise pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    np.savez(
        _data_path("cifar10"),
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"Dataset saved to {_data_path('cifar10')}")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples:     {len(x_test)}")


# ── configure ────────────────────────────────────────────────────────────────

def configure(args: argparse.Namespace) -> None:
    """Download and pre-process datasets used by the selected models."""

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.dataset in {"mnist", "all"}:
        _configure_mnist()
    if args.dataset in {"cifar10", "all"}:
        _configure_cifar10()


# ── train ────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Train the selected model and save it to disk."""
    model_type: str = args.model
    spec = MODEL_TYPES[model_type]
    data_path = _data_path(spec["dataset"])
    if not os.path.exists(data_path):
        sys.exit(
            f"Error: Dataset not found at {data_path}. "
            f"Run 'configure --dataset {spec['dataset']}' first."
        )

    run = _run_name(model_type, args.tag, args.augment)
    save_path = _model_path(run)
    if os.path.exists(save_path) and not args.overwrite:
        sys.exit(
            f"Error: {save_path} already exists.\n"
            f"       Refusing to overwrite a previous run. Either pass --overwrite "
            f"to replace it,\n       or use --tag NAME to save this run under a "
            f"different name."
        )

    data = np.load(data_path)
    x_train, y_train = data["x_train"], data["y_train"]

    cls = spec["classifier"]
    classifier = cls()
    classifier.model = classifier.build_model()

    cb_list = []
    for name in (args.callbacks or []):
        if name == "early_stopping":
            cb_list.append(tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, verbose=1))
        elif name == "reduce_lr":
            cb_list.append(tf.keras.callbacks.ReduceLROnPlateau(
                patience=3, factor=0.5, verbose=1))

    aug_note = " with augmentation" if args.augment else ""
    print(f"Training {model_type} model on {spec['dataset']}{aug_note} \u2026 (run: {run})")
    history = classifier.train(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                               augment=args.augment, callbacks=cb_list or None)

    os.makedirs(MODEL_DIR, exist_ok=True)
    classifier.save(save_path)
    print(f"Model saved to {save_path}")

    # Persist the learning curves so they survive later runs (Tasks 3 and 4).
    if history is not None and getattr(history, "history", None):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        history_path = _history_path(run)
        payload = {
            "run": run,
            "model": model_type,
            "dataset": spec["dataset"],
            "augment": bool(args.augment),
            "epochs_requested": args.epochs,
            "epochs_completed": len(next(iter(history.history.values()))),
            "batch_size": args.batch_size,
            "callbacks": list(args.callbacks or []),
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"History saved to {history_path}")


# ── test ─────────────────────────────────────────────────────────────────────

def test(args: argparse.Namespace) -> None:
    """Evaluate the selected model and write results to a Markdown file."""
    model_type: str = args.model
    spec = MODEL_TYPES[model_type]
    data_path = _data_path(spec["dataset"])
    if not os.path.exists(data_path):
        sys.exit(
            f"Error: Dataset not found at {data_path}. "
            f"Run 'configure --dataset {spec['dataset']}' first."
        )

    run = _run_name(model_type, args.tag, args.augment)
    model_path = _model_path(run)
    if not os.path.exists(model_path):
        train_hint = f"train --model {model_type}"
        if args.tag:
            train_hint += f" --tag {args.tag}"
        elif args.augment:
            train_hint += " --augment"
        sys.exit(f"Error: Trained model not found at {model_path}. Run '{train_hint}' first.")

    data = np.load(data_path)
    x_test, y_test = data["x_test"], data["y_test"]

    cls = spec["classifier"]
    classifier = cls()
    classifier.load(model_path)

    print(f"Evaluating {model_type} model on {spec['dataset']} … (run: {run})")
    results = classifier.evaluate(x_test, y_test)

    y_pred = results["y_pred"]
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    num_weights = classifier.model.count_params()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Markdown report
    md_path = os.path.join(RESULTS_DIR, f"{run}_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {run.upper()} Model — Test Results\n\n")
        f.write(f"- **Model:** {model_type}\n")
        f.write(f"- **Run:** {run}\n")
        f.write(f"- **Dataset:** {spec['dataset']}\n")
        f.write(f"- **Loss:** {results['loss']:.4f}\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **Precision:** {precision:.4f}\n")
        f.write(f"- **Recall:** {recall:.4f}\n")
        f.write(f"- **F1-score:** {f1:.4f}\n")
        f.write(f"- **Weights:** {num_weights}\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("```\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n```\n")

    # JSON report
    json_path = os.path.join(RESULTS_DIR, f"{run}_results.json")
    json_data = {
        "run": run,
        "model": model_type,
        "augment": bool(args.augment) or run.endswith("_aug"),
        "dataset": spec["dataset"],
        "weights": num_weights,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Results saved to {md_path}")
    print(f"Results saved to {json_path}")


# ── summary ───────────────────────────────────────────────────────────────────

def summary(args: argparse.Namespace) -> None:
    """Read all JSON result files and write a combined summary.md."""
    json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_results.json")))
    if not json_files:
        sys.exit("Error: No result JSON files found. Run 'test' first.")

    records = []
    for path in json_files:
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))

    md_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Image Classification — Summary\n\n")
        f.write("| Run | Model | Dataset | Augment | Weights | Accuracy | Precision | Recall | F1-score |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in records:
            run = r.get("run", r["model"])
            augment = "yes" if r.get("augment") else "no"
            f.write(
                f"| {run} | {r['model']} | {r.get('dataset', 'mnist')} | {augment} "
                f"| {r['weights']:,} | {r['accuracy']:.4f} | {r['precision']:.4f} "
                f"| {r['recall']:.4f} | {r['f1_score']:.4f} |\n"
            )

    print(f"Summary saved to {md_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Image Classification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # configure
    configure_parser = subparsers.add_parser("configure", help="Download and prepare datasets")
    configure_parser.add_argument(
        "--dataset", default="mnist", choices=["mnist", "cifar10", "all"],
        help="Dataset to prepare (default: mnist)",
    )

    # train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model", required=True, choices=MODEL_TYPES.keys(),
        help="Model type to train (logistic | nn | cifar_mlp | cnn)",
    )
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    train_parser.add_argument("--augment", action="store_true",
                              help="Enable real-time data augmentation (Task 3 bonus). "
                                   "Saves under the '<model>_aug' run name so the "
                                   "non-augmented run is preserved.")
    train_parser.add_argument("--callbacks", nargs="*", default=[],
                              choices=["early_stopping", "reduce_lr"],
                              help="Training callbacks (Task 5 bonus)")
    train_parser.add_argument("--tag", default=None,
                              help="Custom run name for the saved model, results and "
                                   "history files (default: the model type, or "
                                   "'<model>_aug' with --augment)")
    train_parser.add_argument("--overwrite", action="store_true",
                              help="Allow overwriting an existing model file for this run")

    # test
    test_parser = subparsers.add_parser("test", help="Evaluate a trained model")
    test_parser.add_argument(
        "--model", required=True, choices=MODEL_TYPES.keys(),
        help="Model type to evaluate (logistic | nn | cifar_mlp | cnn)",
    )
    test_parser.add_argument("--augment", action="store_true",
                             help="Evaluate the augmented run ('<model>_aug') instead "
                                  "of the plain one")
    test_parser.add_argument("--tag", default=None,
                             help="Run name to evaluate (must match the --tag used at "
                                  "train time)")

    # summary
    subparsers.add_parser("summary", help="Generate summary.md from all test results")

    args = parser.parse_args()

    commands = {
        "configure": configure,
        "train": train,
        "test": test,
        "summary": summary,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
