"""
Verification script for the Accuracy `average` parameter fix (issue #513)
===========================================================================

Purpose:
    Confirm the implemented fix in ignite/metrics/accuracy.py actually works:
    - Accuracy(is_multilabel=True, average=False) returns a per-label tensor
    - Values match hand-computed expectations
    - Default behavior (no `average`, or `average=None`) is unchanged
    - Validation errors are raised for invalid `average` usage
    - The metric survives multiple reset()/update() cycles (this is the
      exact scenario that exposed the original bug: `_num_correct` was
      initialized as a real 0-dim tensor in reset(), which crashes on the
      first per-label accumulation because in-place `+=` can't broadcast
      a tensor from shape () to (C,))

Usage:
    conda activate ignite-dev
    python3 verify_fix_513.py
"""

import torch
from ignite.metrics import Accuracy


def make_multilabel_data():
    y_true = torch.tensor([
        [0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1],
    ])
    y_pred = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1],
    ])
    return y_pred, y_true


def check_per_label_output(y_pred, y_true):
    print("Accuracy(is_multilabel=True, average=False)")
    print("-" * 60)
    acc = Accuracy(is_multilabel=True, average=False)
    acc.update((y_pred, y_true))
    result = acc.compute()

    expected = (y_true == y_pred).float().mean(dim=0)
    print(f"  Output type:  {type(result).__name__}")
    print(f"  Output shape: {result.shape}")
    print(f"  Value:        {result}")
    print(f"  Expected:     {expected}")
    assert torch.allclose(result, expected), "Per-label output does not match hand-computed values!"
    print("  OK: matches hand-computed per-label accuracy")
    print()


def check_backward_compatible_default(y_pred, y_true):
    print("Accuracy(is_multilabel=True)  [no `average` -> unchanged subset-accuracy scalar]")
    print("-" * 60)
    acc = Accuracy(is_multilabel=True)
    acc.update((y_pred, y_true))
    result = acc.compute()
    print(f"  Output type: {type(result).__name__}")
    print(f"  Value:       {result}")
    assert isinstance(result, float), "Default multilabel behavior should still return a float!"
    print("  OK: still a plain float, default behavior preserved")
    print()


def check_validation_errors():
    print("Validation errors")
    print("-" * 60)
    try:
        Accuracy(average=False)  # missing is_multilabel=True
        raise AssertionError("Expected ValueError, none was raised")
    except ValueError as e:
        print(f"  OK: Accuracy(average=False) without is_multilabel raised: {e}")

    try:
        Accuracy(average="macro")  # unsupported value
        raise AssertionError("Expected ValueError, none was raised")
    except ValueError as e:
        print(f"  OK: Accuracy(average='macro') raised: {e}")
    print()


def check_multi_epoch_reset(y_pred, y_true):
    # This is the scenario that caught the original bug: reset() must correctly
    # re-initialize the per-label accumulator on every epoch, not just the first.
    print("Multiple reset()/update() cycles (regression check for the reset() bug)")
    print("-" * 60)
    acc = Accuracy(is_multilabel=True, average=False)
    for epoch in range(3):
        acc.reset()
        acc.update((y_pred, y_true))
        result = acc.compute()
        expected = (y_true == y_pred).float().mean(dim=0)
        assert torch.allclose(result, expected), f"Epoch {epoch}: mismatch after reset()/update()"
        print(f"  epoch {epoch}: OK -> {result.tolist()}")
    print()


def main():
    import ignite
    print(f"Using ignite from: {ignite.__file__}")
    print(f"Ignite version:    {ignite.__version__}")
    print()

    y_pred, y_true = make_multilabel_data()
    print(f"y_true:\n{y_true}")
    print(f"y_pred:\n{y_pred}")
    print()

    check_per_label_output(y_pred, y_true)
    check_backward_compatible_default(y_pred, y_true)
    check_validation_errors()
    check_multi_epoch_reset(y_pred, y_true)

    print("=" * 60)
    print("All checks passed.")


if __name__ == "__main__":
    main()
