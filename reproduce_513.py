"""
Reproduction script for pytorch/ignite issue #513
=================================================
Label-wise metrics (Accuracy etc.) for multi-label problems.

Purpose:
    Verify the current state of Accuracy, Precision, and Recall in Ignite's
    master branch to confirm what still needs to be built for issue #513.

    The 2019 issue reported that all three metrics could only produce averaged
    scalars in the multi-label case, with no way to get per-label results.
    Since then, Precision and Recall appear to have been refactored to natively
    support per-label output (via the `average` parameter). Accuracy has not.

    This script confirms that empirically.

Expected findings:
    1. Recall(is_multilabel=True)   -> returns per-label tensor  (fixed)
    2. Precision(is_multilabel=True) -> returns per-label tensor (fixed)
    3. Accuracy(is_multilabel=True) -> returns single scalar    (still needs work)

Usage:
    conda activate ignite-dev
    python3 reproduce_513.py
"""

import torch
from ignite.metrics import Accuracy, Precision, Recall


def make_multilabel_data():
    """
    Small hand-crafted multi-label batch (5 samples, 3 labels each).
    We can compute expected per-label accuracy/precision/recall by hand
    to sanity-check the outputs.
    """
    y_true = torch.tensor([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 0],
    ])
    y_pred = torch.tensor([
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    return y_pred, y_true


def hand_computed_expectations(y_pred, y_true):
    """
    Compute per-label accuracy/precision/recall manually so we can
    verify what the metrics return matches expectation.
    """
    print("Hand-computed per-label expectations")
    print("-" * 60)

    # Per-label accuracy: fraction of samples where prediction == truth
    matches = (y_pred == y_true).float()  # shape: (n_samples, n_labels)
    per_label_acc = matches.mean(dim=0)
    print(f"Per-label accuracy (expected):  {per_label_acc.tolist()}")

    # Per-label precision: TP / (TP + FP), i.e. of the ones we predicted
    # positive, how many were actually positive
    tp = ((y_pred == 1) & (y_true == 1)).sum(dim=0).float()
    fp = ((y_pred == 1) & (y_true == 0)).sum(dim=0).float()
    per_label_prec = tp / (tp + fp + 1e-15)
    print(f"Per-label precision (expected): {per_label_prec.tolist()}")

    # Per-label recall: TP / (TP + FN)
    fn = ((y_pred == 0) & (y_true == 1)).sum(dim=0).float()
    per_label_rec = tp / (tp + fn + 1e-15)
    print(f"Per-label recall (expected):    {per_label_rec.tolist()}")
    print()


def test_recall(y_pred, y_true):
    """
    Recall's current behavior in multilabel mode:
    with average=False (default), it should return a per-label tensor.
    """
    print("Recall(is_multilabel=True)")
    print("-" * 60)
    recall = Recall(is_multilabel=True)  # average=False by default
    recall.update((y_pred, y_true))
    result = recall.compute()
    print(f"  Output type:  {type(result).__name__}")
    print(f"  Output shape: {getattr(result, 'shape', 'scalar')}")
    print(f"  Value:        {result}")
    print()


def test_precision(y_pred, y_true):
    """
    Precision's current behavior in multilabel mode:
    same as Recall, should return per-label tensor by default.
    """
    print("Precision(is_multilabel=True)")
    print("-" * 60)
    precision = Precision(is_multilabel=True)
    precision.update((y_pred, y_true))
    result = precision.compute()
    print(f"  Output type:  {type(result).__name__}")
    print(f"  Output shape: {getattr(result, 'shape', 'scalar')}")
    print(f"  Value:        {result}")
    print()


def test_accuracy(y_pred, y_true):
    """
    Accuracy's current behavior in multilabel mode:
    should return a single scalar with no way to opt into per-label output.
    This is the actual gap that issue #513 still points at.
    """
    print("Accuracy(is_multilabel=True)")
    print("-" * 60)
    accuracy = Accuracy(is_multilabel=True)
    accuracy.update((y_pred, y_true))
    result = accuracy.compute()
    print(f"  Output type:  {type(result).__name__}")
    print(f"  Output shape: {getattr(result, 'shape', 'scalar (Python float)')}")
    print(f"  Value:        {result}")
    print()


def check_accuracy_api():
    """
    Print Accuracy's constructor signature to confirm there is no
    `average` parameter (which is what Precision/Recall got via refactor).
    """
    import inspect
    print("Accuracy.__init__ signature (current master)")
    print("-" * 60)
    sig = inspect.signature(Accuracy.__init__)
    print(f"  {sig}")
    print()


def main():
    print("=" * 60)
    print("Reproducing pytorch/ignite issue #513")
    print("=" * 60)
    print()

    # Show which ignite we're actually running against
    import ignite
    print(f"Using ignite from: {ignite.__file__}")
    print(f"Ignite version:    {ignite.__version__}")
    print()

    y_pred, y_true = make_multilabel_data()
    print(f"y_true:\n{y_true}")
    print(f"y_pred:\n{y_pred}")
    print()

    hand_computed_expectations(y_pred, y_true)
    test_recall(y_pred, y_true)
    test_precision(y_pred, y_true)
    test_accuracy(y_pred, y_true)
    check_accuracy_api()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("If Recall and Precision above returned per-label tensors and")
    print("Accuracy returned a single scalar with no `average` option,")
    print("then issue #513 has shrunk in scope: only Accuracy still needs")
    print("label-wise support.")


if __name__ == "__main__":
    main()