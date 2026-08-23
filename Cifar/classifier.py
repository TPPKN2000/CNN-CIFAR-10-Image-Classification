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
        """Train the model on the given data.

        TODO: Implement this method.
        - If self.model is None, call self.build_model() to create it.
        - Use the model's fit() method with the provided parameters.
        - Pass callbacks to fit() if provided.
        - Return the History object from fit().

        Bonus (Task 3): When augment=True, apply real-time data augmentation
        (e.g., using ImageDataGenerator with horizontal flip and shifts).
        When augment=False (the default), train normally — the self-test
        relies on this default path.
        """
        raise NotImplementedError

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on the test data.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's evaluate() method to get loss and accuracy.
        - Use the model's predict() method and np.argmax to get predicted labels.
        - Return a dict with keys: "loss", "accuracy", "y_pred".
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the model to the given file path.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's save() method.
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load a model from the given file path.

        TODO: Implement this method.
        - Use tf.keras.models.load_model() and assign to self.model.
        """
        raise NotImplementedError


class LogisticRegressionClassifier(MNISTClassifier):
    """Logistic regression (single dense layer with softmax)."""

    def build_model(self) -> tf.keras.Model:
        """Build a logistic regression model for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - A single Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="sgd", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError


class NeuralNetworkClassifier(MNISTClassifier):
    """Simple feedforward neural network."""

    def build_model(self) -> tf.keras.Model:
        """Build a simple neural network for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - Dense hidden layer with 128 units and ReLU activation.
          - Dense hidden layer with 64 units and ReLU activation.
          - Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError


class CIFARMLPClassifier(MNISTClassifier):
    """Baseline MLP for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        """Build a baseline MLP model for CIFAR-10.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 32x32x3 images.
          - Flatten layer.
          - Dense hidden layer with 512 units and ReLU activation.
          - Dense hidden layer with 256 units and ReLU activation.
          - Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError


class CIFARCNNClassifier(MNISTClassifier):
    """Convolutional neural network for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        """Build a CNN model for CIFAR-10.

        TODO: Implement this method.
        - Create a Sequential model with the following blocks:
          - Input layer accepting 32x32x3 images.
          Block 1:
          - Conv2D(32, 3, padding="same", activation="relu") + BatchNormalization
          - Conv2D(32, 3, padding="same", activation="relu") + BatchNormalization
          - MaxPooling2D(2) + Dropout(0.2)
          Block 2:
          - Conv2D(64, 3, padding="same", activation="relu") + BatchNormalization
          - Conv2D(64, 3, padding="same", activation="relu") + BatchNormalization
          - MaxPooling2D(2) + Dropout(0.3)
          Block 3:
          - Conv2D(128, 3, padding="same", activation="relu") + BatchNormalization
          - Conv2D(128, 3, padding="same", activation="relu") + BatchNormalization
          - MaxPooling2D(2) + Dropout(0.4)
          Head:
          - Flatten
          - Dense(256, activation="relu") + BatchNormalization + Dropout(0.5)
          - Dense(10, activation="softmax")
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        raise NotImplementedError


class CIFARCNNImprovedClassifier(MNISTClassifier):
    """Improved CNN for CIFAR-10 (Task 5 — Bonus)."""

    def build_model(self) -> tf.keras.Model:
        """Build an improved CNN model for CIFAR-10.

        TODO (Bonus): Implement an improved version of CIFARCNNClassifier.
        Try at least one of:
        - GlobalAveragePooling2D instead of Flatten
        - Deeper CNN blocks or more filters
        - Different optimizer or learning rate
        - Label smoothing (requires CategoricalCrossentropy + one-hot labels)
        The model must accept (32, 32, 3) input and output 10 softmax probabilities.
        This class is NOT checked by the self-test — any architecture is allowed.
        """
        raise NotImplementedError


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
