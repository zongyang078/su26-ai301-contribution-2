# Contribution 2: Label-wise metrics (Accuracy) for multi-label problems

**Contribution Number:** 2
**Student:** Zongyang Li
**Issue:** https://github.com/pytorch/ignite/issues/513
**Status:** Phase IV [In Progress]

---

## Why I Chose This Issue

I chose this issue because it lives in the same `ignite/metrics/` module as my first contribution (issue #1757, `EpochMetric` output types, PR #3789), so I can build directly on the environment, testing patterns, and code style I already learned, and on my existing relationship with the maintainer (`vfdev-5`).

It's also a genuinely good "second contribution" in a different sense: the issue is five years old, has two prior attempted PRs that both stalled, and currently has an adjacent, unrelated effort (a `TopK` metric wrapper, issues #3568/#3610) touching the same class. Rather than a clean, unclaimed feature request, this is a realistic case of "figure out what's actually still true and actually still needed before writing any code" — which is most of what real open-source contribution work looks like once you're past a first PR.

Technically, this matches my interest in evaluation/metrics engineering for ML systems — extending accumulator-style classes (`reset`/`update`/`compute` lifecycle) correctly, handling tensor-shape edge cases, and understanding distributed synchronization (`sync_all_reduce`) are all directly relevant to the AI/ML engineering roles I'm applying for.

---

## Understanding the Issue

### Problem Description
In multi-label classification (each sample can belong to multiple, non-mutually-exclusive classes, e.g. `y = [0, 1, 1]`), the original 2019 issue asked for label-wise output from Ignite's `Accuracy`, `Precision`, and `Recall` — a per-label tensor instead of a single averaged scalar, so users can diagnose which specific label is driving overall model error (the issue author's own example: "a 70% accuracy could be a 30% error in a single label, or a more modest error scattered across 3 labels").

Since 2019 the metrics module has evolved significantly, so before writing any code I re-established the actual current scope against the live codebase (v0.6.0) rather than trusting the 2019 issue description at face value.

### Reduced Scope After Investigation

Reproduction against `master` (see `reproduce_513.py`) confirms two of the three metrics originally targeted already support this natively:

| Metric | Multi-label per-label output in v0.6.0? | Notes |
|---|---|---|
| `Precision(is_multilabel=True)` | ✅ Yes | Returns `tensor([0.4000, 0.6667, 0.0000])` on my reproduction batch — matches hand-computed values exactly. Also supports `average='micro'/'macro'/'weighted'/'samples'`. |
| `Recall(is_multilabel=True)` | ✅ Yes | Returns `tensor([1., 1., 0.])` — matches hand-computed values exactly. Same aggregation modes as Precision. |
| `Accuracy(is_multilabel=True)` | ❌ No | Constructor has no `average` parameter. `compute()` always returns a Python float scalar. |

**Root cause:** `Precision` and `Recall` share a `_BasePrecisionRecall` base class that was refactored at some point after 2019 to support the multiple `average` modes. `Accuracy` was never migrated to a parallel structure and still runs the original 2019-era logic.

### The Actual Remaining Gap
Only `Accuracy` still needs label-wise support — the scope has shrunk from 3 classes to 1.

Reproduction also surfaced a related, more subtle problem: current multi-label `Accuracy` implements **subset accuracy** semantics — a sample only counts as correct if its *entire* label vector matches exactly. On my reproduction batch, 9 of 15 individual label predictions were correct (60% element-wise), but `Accuracy` returned `0.0`, because not a single one of the 5 rows matched exactly. This is technically consistent with `sklearn.metrics.accuracy_score`'s documented default multi-label behavior, but it is a well-known source of user confusion, and it's exactly the failure mode the original issue author was complaining about — with no way to see label-wise scores, a user can't even fall back to something as simple as "average the element-wise correctness."

I also read the actual source (`ignite/metrics/accuracy.py`) rather than reasoning about it from the issue thread alone, and found the precise reason a naive fix isn't a copy-paste of `Precision`/`Recall`'s pattern — see **Solution Approach → Analysis** below.

### Expected Behavior (after fix)
- `Accuracy(is_multilabel=True, average=False)` → returns a per-label tensor of shape `(num_labels,)`
- API surface aligned with `Precision`/`Recall`
- Existing default behavior (`is_multilabel=False`, or `is_multilabel=True` without `average`) preserved exactly, for backward compatibility

### Affected Components
- `ignite/metrics/accuracy.py` — primary changes to `Accuracy`, possibly shared logic in `_BaseClassification`
- `tests/ignite/metrics/test_accuracy.py` — new tests for label-wise behavior, modeled on the existing multilabel + `average` tests in `tests/ignite/metrics/test_precision.py`
- Docstrings + `docs/source/metrics.rst` — usage example for the new mode

---

## Reproduction Process

### Environment Setup
1. Activated the existing `ignite-dev` conda environment (Python 3.10, PyTorch 2.12) that I set up for my first contribution — confirmed active via the `(ignite-dev)` prefix in the terminal prompt.
2. Verified the environment's `ignite` package actually resolves to my fork's editable install, not a pip-installed release, by running:
   ```bash
   python3 -c "import ignite; print(ignite.__file__)"
   ```
   Output: `/Users/lizongyang/code/AI301/ignite/ignite/__init__.py` — confirms it points at my local fork directory, so any code changes I make will actually be picked up when I re-run verification, rather than silently testing against a stale installed copy.
3. Synced my existing fork (`zongyang078/ignite`, originally created for Contribution 1) with `upstream/master`, since it had been several months since my last contribution:
   ```bash
   cd /Users/lizongyang/code/AI301/ignite
   git remote -v          # confirmed `upstream` remote points to pytorch/ignite
   git fetch upstream
   git checkout master
   git merge upstream/master
   git push origin master
   ```
   This sync is what pulled in the `_BasePrecisionRecall` refactor discussed below — without it, my reproduction script would have been running against a stale version of `Precision`/`Recall` and could have produced misleading results about what's already fixed.
4. No new packages needed — `torch`, `ignite` (editable install), and everything else required was already present from Contribution 1's setup.

### Steps to Reproduce
1. Construct a small hand-crafted multi-label batch (5 samples × 3 labels, 0/1 values).
2. Hand-compute per-label accuracy, precision, and recall to establish ground truth independently.
3. Run `Accuracy`, `Precision`, and `Recall` with `is_multilabel=True` and compare against the hand-computed values.
4. Inspect `Accuracy.__init__`'s signature directly to confirm there's no `average` parameter.

Full script: `reproduce_513.py` in this repo.

### Reproduction Evidence

- **Commit showing reproduction:** [`reproduce_513.py`](https://github.com/zongyang078/su26-ai301-contribution-2) in my notes repo (hand-crafted 5×3 multilabel batch with hand-computed ground truth)
- **Screenshots/logs:**

  Hand-computed expectations (5 samples × 3 labels):
  - Per-label accuracy: `[0.4, 0.8, 0.0]`
  - Per-label precision: `[0.4, 0.6667, 0.0]`
  - Per-label recall: `[1.0, 1.0, 0.0]`

  Actual outputs from `master`:

  | Call | Output | Shape | Matches hand-calc? |
  |---|---|---|---|
  | `Recall(is_multilabel=True).compute()` | `tensor([1., 1., 0.])` | `(3,)` | ✅ |
  | `Precision(is_multilabel=True).compute()` | `tensor([0.4000, 0.6667, 0.0000])` | `(3,)` | ✅ |
  | `Accuracy(is_multilabel=True).compute()` | `0.0` | scalar (Python float) | N/A — no per-label option exists |

  `Accuracy.__init__` signature (confirmed via `inspect.signature`, v0.6.0):
  ```
  (self, output_transform, is_multilabel: bool = False, device, skip_unrolling: bool = False)
  ```
  No `average` parameter — confirms the API gap versus `Precision`/`Recall`.

- **My findings:**
  1. Two of the three metrics originally targeted by issue #513 (`Precision`, `Recall`) have already been fixed by a prior, undocumented-in-the-issue refactor (`_BasePrecisionRecall`).
  2. `Accuracy` was not migrated and still lacks label-wise support entirely.
  3. Beyond the missing feature, current multi-label `Accuracy` uses subset-accuracy semantics — correct per `sklearn` convention, but the exact UX pitfall the original issue author was reacting to, since it means a model with meaningfully useful per-label performance can still show `0.0` overall.

---

## Solution Approach

### Analysis

I read `accuracy.py` and `precision.py` side by side rather than assuming the fix is a drop-in copy of the `Precision`/`Recall` pattern, and found a structural difference that matters for implementation:

**`Precision`/`Recall`'s per-label output is "free"** — their underlying `correct` computation (`_BasePrecisionRecall._prepare_output`) is already elementwise (`correct = y * y_pred`), so per-label results just mean *not collapsing* the label dimension when summing:
```python
# precision.py / recall.py, when average in [False, None, 'macro', 'weighted']
self._denominator += y_pred.sum(dim=0)   # shape (C,) — stays per-label
self._numerator += correct.sum(dim=0)    # shape (C,) — stays per-label
```

**`Accuracy`'s current multi-label `correct` is fundamentally different** — it uses `torch.all(dim=-1)`, which collapses the label dimension by design (subset accuracy):
```python
# accuracy.py, current
correct = torch.all(y == y_pred.type_as(y), dim=-1)  # label dim already destroyed here
```

**Implication:** you can't get per-label output out of the *existing* `correct` tensor — by the time it's computed, the label dimension is gone. Getting label-wise `Accuracy` requires computing a **different, elementwise `correct`** (`y == y_pred`, without the `torch.all` collapse) for the new mode, coexisting with the current subset-accuracy path (which must remain the default, unchanged, for backward compatibility).

There's a second, more subtle asymmetry worth flagging explicitly rather than discovering it in review: for `Precision`/`Recall`, the per-label denominator varies by label (predicted-positive or actual-positive count differs per label), which is why `'weighted'`/`'micro'` are meaningful there. For `Accuracy`, the per-label denominator is just `N` (total samples) — identical for every label — so the full `average` vocabulary doesn't map onto `Accuracy` with unchanged meaning. This needs an explicit maintainer decision (see Open Questions below), not a silent assumption.

I've written up the full historical and source-level analysis, including the complete timeline of prior attempts on this issue (2019 PR #516, PR #542) and the currently-stalled adjacent effort (#3568/#3610, a `TopK` wrapper that also touches `Accuracy`), in a separate document: **`RELATED_WORK.md`** in this repo.

### Proposed Solution
Add an `average` parameter to `Accuracy`, aligning its API with `Precision`/`Recall`, scoped narrowly:
- `average=None` (default) — preserve current subset-accuracy scalar behavior exactly (backward compatible)
- `average=False` — return per-label tensor in multi-label case, using a new elementwise `correct` computation
- Raise `ValueError` if `average` is set without `is_multilabel=True`
- Explicitly **not** touching `Accuracy`'s internal structure (e.g. not adding a `_prepare_output` method) so this stays orthogonal to the unrelated, currently-unresolved `#3610` TopK wrapper effort

Whether to support the fuller `average` vocabulary (`'micro'`/`'macro'`/`'weighted'`/`'samples'`) beyond `False` is an open question for the maintainer — see below — since those modes don't have unambiguous meaning for `Accuracy` the way they do for `Precision`/`Recall` (see Analysis above).

### Open Questions for Maintainer
Rather than deciding these unilaterally, I plan to ask directly in my check-in comment:
1. Is this narrow scope (just `average=False`) acceptable as a first PR, or should this wait for `#3610` to land so `Accuracy` can be refactored in one pass?
2. Should the initial implementation support the full `average` vocabulary, or just `False`? If the former, what should `'weighted'`/`'micro'`/`'samples'` mean for `Accuracy` specifically?
3. Since `Accuracy`'s current default can't become per-label without a breaking change (unlike `Precision`/`Recall`, where `False` was already the default), is `average=None` an acceptable default that preserves current behavior, with `average=False` as the explicit opt-in?

### Implementation Plan
Using UMPIRE framework (adapted):

**Understand:** Users need per-label accuracy in multi-label settings; `Accuracy` is the only one of the three original metrics still lacking it, and its current subset-accuracy default is a known UX pitfall.

**Match:** `_BasePrecisionRecall` in `precision.py`/`recall.py` is the reference pattern for accumulating and returning per-label tensors, `sync_all_reduce`-compatible. My prior PR (#3789, `EpochMetric` output types) already established the testing/validation conventions I reused.

**Plan:**
1. Confirm scope with maintainer — posted check-in comment; proceeded with Scope A (additive, no breaking changes)
2. Add elementwise `correct` computation to `Accuracy.update()`'s multilabel path
3. Change `_num_correct` from always-tensor to `int | torch.Tensor` (mirrors `Precision`/`Recall`)
4. Add input validation (`average` set without `is_multilabel=True` → `ValueError`)
5. Verify `sync_all_reduce` handles the tensor case in distributed settings
6. Add tests (per-label values, edge cases, backward compat, Engine integration, distributed)
7. Update docstring with runnable multilabel per-label example

**Implement:** Branch [`feat/accuracy-average-param`](https://github.com/zongyang078/ignite/tree/feat/accuracy-average-param) — see [PR #3810](https://github.com/pytorch/ignite/pull/3810) for the full diff. Key implementation detail: `reset()` had to change from `torch.tensor(0)` to plain `int 0` because PyTorch in-place `+=` can't broadcast a 0-dim tensor to shape `(C,)` — same `int | torch.Tensor` pattern `Precision`/`Recall` already use.

**Review:** Self-review checklist:
- [x] Backward compatible — default behavior unchanged
- [x] API naming aligned with `Precision`/`Recall` conventions
- [x] Distributed sync (`sync_all_reduce`) verified for tensor `_num_correct`
- [x] Tests cover per-label values, edge cases, backward compat, and validation errors
- [x] Docstring includes a runnable example
- [x] PR description explicitly states scope boundary relative to #3610/#3568

**Evaluate:** `verify_fix_513.py` matches hand-computed values exactly. Full test suite: 363 passed, zero regressions. Also ran `pyrefly check` (the project's CI type checker) — zero errors.

---

## Testing Strategy

### Unit Tests
- [x] `Accuracy(is_multilabel=True, average=False)` returns correct per-label tensor on hand-computed toy dataset (values above)
- [x] Per-label output shape matches `(num_labels,)`
- [x] Edge case: a label with zero positive samples in the batch
- [x] Edge case: all predictions correct → all-ones tensor
- [x] Edge case: all predictions wrong → all-zeros tensor
- [x] `average` set without `is_multilabel=True` raises `ValueError`
- [x] Backward compatibility: `Accuracy()` (binary/multiclass, no `is_multilabel`) unchanged
- [x] Backward compatibility: `Accuracy(is_multilabel=True)` with no `average` still returns the current subset-accuracy scalar, unchanged

### Integration Tests
- [x] Metric attached to an `Engine`, run over multiple epochs, correctly resets/accumulates per-label tensor state each epoch (this is the test that would have caught the `reset()` bug directly)
- [x] Distributed (multi-process) synchronization produces correct per-label results via `sync_all_reduce`

### Manual Testing
Ran the implemented metric against `reproduce_513.py` and a new `verify_fix_513.py` (per-label output, backward-compat default, validation errors, and a multi-epoch reset/update regression check) — all outputs match hand-computed values. Also ran `pytest tests/ignite/metrics/test_accuracy.py -v` (363 passed) plus the wider set of files that touch `Accuracy` before opening the PR.

---

## Implementation Notes

### Week 1 — Phase 1 (Issue Selection)

- Evaluated ~15 candidate issues across vllm-omni, vllm, scikit-learn, pytorch/executorch, vllm-metal, and pytorch/ignite. Each was rejected for a specific reason (already claimed, PR in flight, hardware-inaccessible, design unresolved).
- Selected ignite #513 for its fit with my existing contribution context (same `ignite/metrics/` module, same maintainer, same dev environment as Contribution 1).

### Week 2 — Phase 2 (Reproduce & Plan)

- **Reproduction:** Wrote and ran `reproduce_513.py` against current `master` — discovered that 2 of 3 originally-targeted metrics had already been fixed by an undocumented refactor (`_BasePrecisionRecall`), narrowing the real scope from 3 classes to 1.
- **Historical analysis:** Read PR #516, PR #542, issue #467, issue #3568, and draft PR #3610 end-to-end. Found that both prior attempts stalled on the same API design objection (boolean `labelwise` flag rejected by maintainer), and that an adjacent TopK wrapper effort (#3610) is stalled with an unresolved 3-way design disagreement but doesn't block this work. Full writeup in `RELATED_WORK.md`.
- **Source-level analysis:** Read `accuracy.py` and `precision.py` side-by-side to find the structural reason a naive "copy the Precision pattern" fix wouldn't work (subset accuracy via `torch.all` destroys the label dimension — see Solution Approach → Analysis).
- **Plan:** Scoped the fix to Scope A (additive `average=False`, no breaking changes, orthogonal to #3610). Drafted check-in comment for the issue.

### Week 3 — Phase 3 (Build) + Phase 4 (Submit & Iterate)

- Posted check-in comment to issue #513, then started implementation immediately without waiting for a reply — Scope A is additive and backward-compatible, so the risk of proceeding is low (unlike PR #516/#542 which proposed a new flag the maintainer actively disliked).
- Implemented the `average` parameter: `__init__` validation, elementwise per-label `correct` path in `update()`, tensor-aware `compute()`.
- Found and fixed a runtime bug: `reset()` initialized `_num_correct` as `torch.tensor(0)` (a real 0-dim tensor), which crashes on in-place `+=` with a `(C,)` vector. Fixed by switching to plain `int 0`, matching `Precision`/`Recall`'s `int | torch.Tensor` pattern.
- That fix cascaded into two pre-existing tests (`test_accumulator_device`, `test_integration_batchwise`) that assumed `_num_correct` is always a tensor — updated both.
- Caught a static type error via `pyrefly check` (the project's CI type checker): `int` has no `.item()`. Fixed with `cast(torch.Tensor, ...)`, matching `Precision`'s pattern.
- Wrote comprehensive tests: per-label correctness, edge cases (all-correct, all-wrong, no-positive-label), multi-batch accumulation, Engine integration with multi-epoch reset, distributed integration.
- Verified: 363 tests passed, zero regressions, zero type errors.
- Opened [PR #3810](https://github.com/pytorch/ignite/pull/3810) on 2026-07-15. Awaiting maintainer review.

### Code Changes
- **Files modified:**
  - `ignite/metrics/accuracy.py` — `average` parameter, per-label `update()` path, tensor-aware `compute()`, `cast()` for the type checker
  - `tests/ignite/metrics/test_accuracy.py` — validation test, per-label correctness + edge cases, Engine integration (multi-epoch), distributed integration, `test_accumulator_device` fix
  - `tests/ignite/metrics/test_running_average.py` — type-agnostic `_num_correct` comparison fix
- **Key commits:** on branch `feat/accuracy-average-param` in my fork (`zongyang078/ignite`); see [PR #3810](https://github.com/pytorch/ignite/pull/3810) for the full commit list
- **Approach decisions:**
  - `average=None` default (not `False`) — unlike `Precision`/`Recall`, `Accuracy`'s default can't become per-label without breaking every existing caller (see RELATED_WORK.md §4.5)
  - Only `None`/`False` accepted for now (not the full `'micro'`/`'macro'`/`'weighted'`/`'samples'` vocabulary) — deferred to maintainer input since those don't have unambiguous meaning for `Accuracy` (RELATED_WORK.md §4.4)
  - `_num_correct: int | torch.Tensor` instead of a separate init path per `average` value — mirrors `Precision`/`Recall`'s existing `_numerator`/`_denominator` pattern exactly, so the accumulator "grows into" the right shape on first `update()`

---

## Pull Request

**PR Link:** [PR #3810](https://github.com/pytorch/ignite/pull/3810)

**PR Description:** Adapted from the Proposed Solution / Solution Approach sections above, using the community's actual PR template format (`Fixes #513` / `## Description` / checklist) rather than the longer draft originally written here — explicitly calls out the scope boundary relative to #3568/#3610 and the two incidental test fixes.

**Maintainer Feedback:**
- 2026-07-15: PR opened, awaiting review.

**Status:** Awaiting review

---

## Learnings & Reflections

### Technical Skills Gained
- Reading two implementations of a similar pattern (`_BaseClassification` vs `_BasePrecisionRecall`) side by side to find a structural difference invisible from the issue discussion alone — the distinction between "elementwise `correct`, naturally decomposable per label" and "subset `correct` via `torch.all`, label dimension already collapsed."
- Understanding PyTorch in-place operator semantics: `tensor += vector` fails when shapes don't match because `__iadd__` calls `add_` which can't reallocate, while `int += vector` works because Python falls back to `__radd__` which creates a new object.
- Running a project's actual CI toolchain locally (`pyrefly check`, not just `pytest`) to catch type errors before the PR hits CI — and understanding why `int | torch.Tensor` requires `cast()` for the type checker even when the runtime would be fine.

### Challenges Overcome
- The 2019 issue description was stale — 2 of 3 targeted metrics had been quietly fixed. Confirming this required running code against `master`, not just reading the discussion. Lesson: always reproduce before planning.
- Distinguishing a blocking dependency from an adjacent one: the #3610 TopK wrapper effort touches `Accuracy` but doesn't solve or block this issue's ask. Recognizing that required reading its full design debate (three competing proposals, still unresolved) rather than assuming "same class = same effort."
- The `reset()` bug was invisible from reading the diff — it only surfaced at runtime on the second epoch. Writing a multi-epoch regression test (`check_multi_epoch_reset`) before considering the implementation "done" is what caught it.

### What I'd Do Differently Next Time
- Run the reproduction script earlier in the process — the scope reduction it revealed reshaped every subsequent decision, and I could have deprioritized some of the tangential historical reading until after confirming what was actually still relevant.
- Check the project's CI pipeline configuration (`.github/workflows/`) before opening the PR, not after — I found the `pyrefly check` requirement late and had to do a follow-up fix commit.

---

## Resources Used
- Original issue: https://github.com/pytorch/ignite/issues/513
- Prior attempt #1 (2019, stalled on API design): https://github.com/pytorch/ignite/pull/516
- Prior attempt #2 (2019, stalled same day as maintainer feedback): https://github.com/pytorch/ignite/pull/542
- Related issue (Top-K precision/recall): https://github.com/pytorch/ignite/issues/467
- Adjacent effort (TopK wrapper): https://github.com/pytorch/ignite/issues/3568 and https://github.com/pytorch/ignite/pull/3610
- Merged precedent for wrapper pattern: https://github.com/pytorch/ignite/pull/3627
- Ignite metrics documentation: https://pytorch.org/ignite/metrics.html
- Full related-work analysis (this repo): `RELATED_WORK.md`
- Reproduction script (this repo): `reproduce_513.py`
- Verification script (this repo): `verify_fix_513.py`
