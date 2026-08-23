# Lab Assignment 2: CNN for CIFAR-10 Image Classification

## 1. Objective

In this extension lab, you will move beyond MNIST and build a **Convolutional Neural Network (CNN)** for the **CIFAR-10** dataset. Unlike MNIST (28x28 grayscale digits), CIFAR-10 contains **32x32 RGB natural images** across 10 object classes, making it significantly more challenging.

You will:

1. Build a baseline fully connected model for CIFAR-10.
2. Build a CNN model and compare performance.
3. Improve generalization using regularization and data augmentation.
4. Analyze class-wise errors and propose architecture improvements.

---

## 2. Why CIFAR-10 Is Challenging

- Images are color (3 channels), not grayscale.
- Classes are visually more diverse (e.g., cat vs dog, automobile vs truck).
- Objects appear with different backgrounds, poses, and scales.
- Decision boundaries are less linearly separable.

Expected result: a simple MLP that works well on MNIST will perform much worse on CIFAR-10, while CNNs should provide a clear gain.

---

## 3. Dataset and Setup

Use pre-processed CIFAR-10 dataset, `data/cifar10.npz` (inside `cifar10.7z`). Training and test sets are fixed for consistent comparison.

### Files you are given

```
Lab3-Cifar/
├── assignment.md            # This document
├── classifier.py            # Stub – YOU IMPLEMENT THIS
├── main.py                  # Training/testing pipeline – do not modify
├── data/cifar10.npz         # Pre-processed dataset (extract from cifar10.7z)
├── models/                  # Created by 'train'
└── results/                 # Created by 'train' and 'test'
```

| File | Role |
|---|---|
| `classifier.py` | Contains the base class `MNISTClassifier` and the CIFAR-10 model classes with method stubs, plus the autograde self-test block at the bottom. **This is the file you will implement**, and the only `.py` file you submit. |
| `main.py` | The complete pipeline — `configure`, `train`, `test`, `summary`. It imports your `classifier.py`. Read it to understand how your methods are called, but do not modify or submit it. |

Install dependencies:

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

> **Note:** TensorFlow 2.15.x is recommended for Python 3.9. For Python 3.10+, TensorFlow 2.16+ should work.

### Run names — how artifacts are stored

Every `train` run is stored under a **run name**, and `test` writes its results under
the same name:

| Run name | Model file | Results | History |
|---|---|---|---|
| `cnn` | `models/cnn_model.keras` | `results/cnn_results.{json,md}` | `results/cnn_history.json` |
| `cnn_aug` | `models/cnn_aug_model.keras` | `results/cnn_aug_results.{json,md}` | `results/cnn_aug_history.json` |

The run name defaults to the model type, becomes `<model>_aug` when you pass
`--augment`, and can be set to anything with `--tag NAME`. This means **Task 3 does
not overwrite your Task 2 model or results** — the two runs live side by side and
both appear in `summary.md`.

Two related conveniences:

- `train` refuses to overwrite an existing model file. Pass `--overwrite` to replace
  it deliberately, or `--tag NAME` to keep both.
- `train` writes `results/<run>_history.json` containing the per-epoch loss and
  accuracy for train and validation. Use these files to plot the curves for Tasks 3
  and 4 — you do not need to re-train to redraw a plot.

---

## 4. Tasks

### Task 1 - Baseline MLP on CIFAR-10

Implement and train a baseline MLP (`CIFARMLPClassifier` in `classifier.py`):

- Input: 32x32x3 image, flattened to 3072 **inside the model** (first layer is `Flatten`)
- Hidden layers: Dense(512, relu), Dense(256, relu)
- Output: Dense(10, softmax)
- Loss: sparse_categorical_crossentropy
- Optimizer: adam

Record test accuracy, parameter count, and confusion matrix.

```bash
python main.py train --model cifar_mlp --epochs 20
python main.py test --model cifar_mlp
```

### Task 2 - CNN Model

Implement and train a CNN (`CIFARCNNClassifier` in `classifier.py`). Use three
convolutional blocks, each of them two `padding="same"` convolutions with
BatchNormalization, then pooling and dropout:

- **Block 1:** Conv2D(32, 3, same, relu) + BN, Conv2D(32, 3, same, relu) + BN, MaxPooling2D(2), Dropout(0.2)
- **Block 2:** Conv2D(64, 3, same, relu) + BN, Conv2D(64, 3, same, relu) + BN, MaxPooling2D(2), Dropout(0.3)
- **Block 3:** Conv2D(128, 3, same, relu) + BN, Conv2D(128, 3, same, relu) + BN, MaxPooling2D(2), Dropout(0.4)
- **Head:** Flatten, Dense(256, relu) + BN + Dropout(0.5), Dense(10, softmax)

Compile with `adam` and `sparse_categorical_crossentropy`.

The docstrings in `classifier.py` give the same architecture layer by layer — follow
them if in doubt.

Train and evaluate:

```bash
python main.py train --model cnn --epochs 20
python main.py test --model cnn
```

### Task 3 - Data Augmentation (Bonus — up to 10 points)

Implement data augmentation in the `train()` method. When `augment=True`, apply real-time image augmentation (e.g., horizontal flip, width/height shift) using `ImageDataGenerator` or Keras preprocessing layers. When `augment=False` (the default), `train()` must behave exactly as before.

Train the CNN with augmentation enabled. Note the `--augment` flag on **both**
commands — it routes this experiment to the `cnn_aug` run name so your Task 2
artifacts stay intact:

```bash
python main.py train --model cnn --augment --epochs 30
python main.py test --model cnn --augment
```

This produces `results/cnn_aug_results.json` and `results/cnn_aug_history.json`
alongside the Task 2 files. Compare the CNN accuracy/loss curves with and without
augmentation by plotting `cnn_history.json` against `cnn_aug_history.json`, and
discuss overfitting reduction in your report.

> **Do not** run the augmented training without `--augment` on the `test` command —
> you would then be evaluating your Task 2 model and reporting it as the augmented
> result.

> **Self-test safety:** The self-test calls `train()` without `augment`, so the default code path is never affected.

### Task 4 - Learning Curves and Error Analysis

For MLP and CNN models:

1. Plot training/validation loss and accuracy curves, using the saved
   `results/cifar_mlp_history.json` and `results/cnn_history.json`.
2. Generate confusion matrix and classification report (the confusion matrix is
   printed into `results/<run>_results.md` by `main.py test`).
3. Identify top 3 most confused class pairs and explain why.

### Task 5 - Model Improvement Challenge (Bonus — up to 10 points)

Implement `CIFARCNNImprovedClassifier` in `classifier.py` with at least **one** architectural or training improvement over the base CNN. Ideas:

- GlobalAveragePooling2D instead of Flatten
- Deeper CNN blocks or more filters
- Different optimizer or learning rate
- Label smoothing (requires switching to `CategoricalCrossentropy` + one-hot labels)

You can also use built-in training callbacks via the `--callbacks` flag:

```bash
python main.py train --model cnn_improved --epochs 50 --callbacks early_stopping reduce_lr
python main.py test --model cnn_improved
```

Available callbacks: `early_stopping` (patience=5, restore best weights), `reduce_lr` (patience=3, factor=0.5).

If you want to try several configurations and keep them all, give each one its own
run name with `--tag`:

```bash
python main.py train --model cnn_improved --tag improved_gap --epochs 50 --callbacks early_stopping
python main.py test  --model cnn_improved --tag improved_gap
```

Only `cnn_improved_results.json` (the untagged run) is collected for grading, so run
your final configuration without `--tag`.

> **Self-test safety:** `CIFARCNNImprovedClassifier` is **not** checked by the self-test. You can design any architecture as long as it accepts `(32, 32, 3)` input and outputs 10 softmax probabilities.

In your report, state your hypothesis **before** training and whether the result supports it.

---

## 5. Submission

- **File name**: `<StudentID>.zip` (e.g., `25127000.zip`)
- **Structure**: The zip file must contain a folder named with your Student ID, with the following files inside (names must match **exactly**):

```
25127000.zip
└── 25127000/
    ├── classifier.py                    # Your completed implementation (all classes)
    ├── report.pdf                       # Your written analysis (2–4 pages, PDF)
    ├── results/
    │   ├── summary.md                   # Generated by: python main.py summary
    │   ├── cifar_mlp_results.json       # Generated by: python main.py test --model cifar_mlp
    │   ├── cifar_mlp_history.json       # Generated by: python main.py train --model cifar_mlp
    │   ├── cnn_results.json             # Generated by: python main.py test --model cnn
    │   ├── cnn_history.json             # Generated by: python main.py train --model cnn
    │   ├── cnn_aug_results.json         # Bonus (Task 3): test --model cnn --augment
    │   ├── cnn_aug_history.json         # Bonus (Task 3): train --model cnn --augment
    │   └── cnn_improved_results.json    # Bonus (Task 5): python main.py test --model cnn_improved
    └── figures/
        ├── learning_curves.png          # Training/validation curves (MLP vs CNN)
        └── confusion_matrix.png         # CNN confusion matrix on the test set
```

**Do NOT include** trained model files (`.keras`), `data/`, `main.py`, or `__pycache__/` in the zip — they are large and are not graded. Submissions with extra files receive a penalty.

The `cnn_aug_*.json` files are only required if you attempt Task 3, and
`cnn_improved_results.json` only if you attempt Task 5; the `_history.json` files for
those bonus runs are what let the grader verify the before/after curves you show in
`report.pdf`. Extra `--tag` runs beyond these are **not** counted as extra files.

---

## 6. Automated Grading

Your submission is graded automatically:

1. **Structure check** — the grader verifies every file above exists at the exact path. Missing files lose their points; misplaced files (right name, wrong folder) incur up to a 10% penalty; extra files up to 5%.
2. **Self-test execution** — the grader runs `python classifier.py`, which executes the self-test block at the bottom of the file (**do not modify it**). It builds `CIFARMLPClassifier` and `CIFARCNNClassifier` on random data and checks the MLP parameter count, the CNN structure (≥ 4 Conv2D, pooling, dropout), softmax outputs, and your `train` / `evaluate` / `save` / `load` implementations — no dataset needed, it finishes in seconds. Both models must accept input of shape `(32, 32, 3)`; the MLP flattens inside the model. **A failing self-test results in 0 from the autograder**, and the submission is flagged for manual review.

Run the self-test yourself before submitting:

```bash
python classifier.py
# Expected: [PASS] cifar_mlp, [PASS] cnn, "Self-test passed." and exit code 0
```

### Grading Rubric

| Component | Points |
|---|---|
| Task 1 — MLP baseline (self-test pass + `cifar_mlp_results.json`) | 20 |
| Task 2 — CNN (self-test pass + `cnn_results.json`) | 30 |
| Task 4 — Learning curves, confusion matrix, error analysis | 20 |
| Report — questions 1, 2, 5 | 20 |
| Submission structure and formatting | 10 |
| **Total** | **100** |
| Task 3 — Augmentation bonus (report + `cnn_aug_results.json`) | +10 |
| Task 5 — Improved model bonus (report + `cnn_improved_results.json`) | +10 |

---

## 7. Report Questions

Answer clearly:

1. How much does CNN improve over MLP on CIFAR-10 in accuracy and F1?
2. Which classes are hardest, and what visual factors may explain this?
3. *(Bonus — answer if you completed Task 3.)* Did augmentation reduce overfitting? Show evidence from learning curves.
4. *(Bonus — answer if you completed Task 5.)* Which improvement did you try, and did the result support your hypothesis?
5. If you had 2x more training time, what would you try next?

---

## 8. Suggested Reference Targets

Typical results with reasonable training settings:

- Baseline MLP: ~45% to 55% test accuracy
- Basic CNN (without augmentation): ~70% to 78%
- CNN + augmentation + callbacks: ~75% to 83%

Exact values may vary by hardware, epochs, and random seed.