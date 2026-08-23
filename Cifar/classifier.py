from __future__ import annotations

import abc
from typing import Optional

import numpy as np
import tensorflow as tf


class MNISTClassifier(abc.ABC):
    """Base class for MNIST digit classifiers."""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None

    @abc.abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Build and return a compiled Keras model."""

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 10, batch_size: int = 128,
              validation_split: float = 0.1,
              augment: bool = False,
              callbacks: list = None) -> tf.keras.callbacks.History:
        """Train the model on the given data."""
        if self.model is None:
            self.model = self.build_model()

        if augment:
            # Real-time augmentation: horizontal flip + width/height shift.
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                validation_split=validation_split,
            )
            train_gen = datagen.flow(
                x_train, y_train, batch_size=batch_size, subset="training"
            )
            val_gen = datagen.flow(
                x_train, y_train, batch_size=batch_size, subset="validation"
            )
            history = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks,
            )
        else:
            history = self.model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
            )
        return history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on the test data."""
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded yet.")
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        probs = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        return {"loss": loss, "accuracy": accuracy, "y_pred": y_pred}

    def save(self, path: str) -> None:
        """Save the model to the given file path."""
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded yet.")
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load a model from the given file path."""
        self.model = tf.keras.models.load_model(path)


class LogisticRegressionClassifier(MNISTClassifier):
    """Logistic regression (single dense layer with softmax)."""

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class NeuralNetworkClassifier(MNISTClassifier):
    """Simple feedforward neural network."""

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class CIFARMLPClassifier(MNISTClassifier):
    """Baseline MLP for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class CIFARCNNClassifier(MNISTClassifier):
    """Convolutional neural network for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),

            # Block 1
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.2),

            # Block 2
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.3),

            # Block 3
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.4),

            # Head
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class CIFARCNNImprovedClassifier(MNISTClassifier):
    """Improved CNN for CIFAR-10 (Task 5 — Bonus).

    Improvements over CIFARCNNClassifier:
      1. GlobalAveragePooling2D instead of Flatten before the dense head —
         removes the largest, most overfitting-prone weight block (a Flatten
         of a 4x4x128 map feeds ~2,000 units into Dense(256); GAP feeds only
         the channel count).
      2. Wider conv blocks (64/128/256 filters instead of 32/64/128) for more
         representational capacity.
      3. A lower Adam learning rate (5e-4) for more stable convergence.

    Hypothesis: this should reduce the train/validation accuracy gap (less
    overfitting) and modestly improve test accuracy versus the baseline CNN.
    """

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),

            # Block 1
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.2),

            # Block 2
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.3),

            # Block 3
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.4),

            # Head — GlobalAveragePooling2D instead of Flatten
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


# ──────────────────────────────────────────────────────────────────────────────
# Autograde self-test — DO NOT MODIFY ANYTHING BELOW THIS LINE
#
# The autograder runs:  python classifier.py
# It builds the CIFAR-10 models and checks your implementation on random data
# (no dataset needed). Exit code 0 = pass. Run it yourself before submitting.
# Note: both models must accept input of shape (32, 32, 3); the MLP flattens
# the image inside the model (first layer), not in the data pipeline.
# ──────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    import os

    rng = np.random.default_rng(0)
    x = rng.random((8, 32, 32, 3), dtype=np.float32)
    y = rng.integers(0, 10, size=(8,))

    def _common_checks(name: str, clf: "MNISTClassifier") -> None:
        probs = clf.model.predict(x, verbose=0)
        assert probs.shape == (8, 10), (
            f"{name}: output shape is {probs.shape}, expected (8, 10) — "
            "the model must accept (32, 32, 3) input"
        )
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), (
            f"{name}: outputs do not sum to 1 — did you use softmax?"
        )
        history = clf.train(x, y, epochs=1, batch_size=4)
        assert history is not None, f"{name}: train() must return the History object"
        result = clf.evaluate(x, y)
        for key in ("loss", "accuracy", "y_pred"):
            assert key in result, f"{name}: evaluate() result is missing key '{key}'"
        assert len(np.asarray(result["y_pred"])) == 8, (
            f"{name}: y_pred must contain one predicted label per sample"
        )
        path = f"_selftest_{name}.keras"
        clf.save(path)
        assert os.path.exists(path), f"{name}: save() did not create {path}"
        clf.load(path)
        assert clf.model is not None, f"{name}: load() did not set self.model"
        os.remove(path)
        print(f"[PASS] {name}")

    # ── Baseline MLP ──
    clf = CIFARMLPClassifier()
    model = clf.build_model()
    assert model is not None, "cifar_mlp: build_model() returned None"
    n_params = model.count_params()
    assert n_params == 1_707_274, (
        f"cifar_mlp: expected 1,707,274 parameters, got {n_params:,}"
        " — check: Flatten, Dense(512), Dense(256), Dense(10)"
    )
    clf.model = model
    _common_checks("cifar_mlp", clf)

    # ── CNN ──
    clf = CIFARCNNClassifier()
    model = clf.build_model()
    assert model is not None, "cnn: build_model() returned None"
    conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    assert len(conv) >= 4, f"cnn: found {len(conv)} Conv2D layers, expected at least 4"
    assert any(isinstance(l, tf.keras.layers.MaxPooling2D) for l in model.layers), (
        "cnn: missing MaxPooling2D layer(s)"
    )
    assert any(isinstance(l, tf.keras.layers.Dropout) for l in model.layers), (
        "cnn: missing Dropout layer(s)"
    )
    clf.model = model
    _common_checks("cnn", clf)

    print("Self-test passed. Your classifier.py is ready to submit.")


if __name__ == "__main__":
    _self_test()
