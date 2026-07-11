# Related Work: Issue #513 (Label-wise metrics for multi-label problems)

**Author:** zongyang078
**Date:** July 2026
**Repo:** pytorch/ignite
**Target issue:** [#513](https://github.com/pytorch/ignite/issues/513)

---

## TL;DR

- Issue #513 (2019) asked for per-label output from `Accuracy`, `Precision`, and `Recall` in multi-label classification.
- **Two of three are already done.** `Precision`/`Recall` got an `average` parameter (`False`/`None`/`'micro'`/`'macro'`/`'weighted'`/`'samples'`) via the `_BasePrecisionRecall` refactor at some point after 2019. Only `Accuracy` was never migrated.
- **Two prior attempts to fix this (PR #516, PR #542) both stalled in 2019** — not on implementation difficulty, but because the maintainer (`vfdev-5`) rejected the proposed API (a `labelwise: bool` flag) and wanted something more unified. That unified vision never landed, and Precision/Recall evolved independently in the meantime using a different (though related) pattern — the `average` string param.
- **A second, unrelated effort is currently touching the same territory**: issues #467/#3568 and draft PR #3610 are trying to build a generic `TopK` metric wrapper, which would also require `Accuracy` to grow new internal structure (`_prepare_output`). That effort has been stalled for ~4 months with an unresolved 3-way design disagreement.
- **The two efforts don't solve the same problem and shouldn't block each other**, but they do touch the same class, so scope needs to be stated explicitly to avoid the appearance of (or actual) collision.
- **Accuracy's per-label semantics are structurally different from Precision/Recall's**, in a way that affects API design (see §4). This is the most important technical finding in this document.

---

## 1. What issue #513 actually asks for

Opened 2019 by `jphdotam`. In multi-label classification (a sample can belong to multiple classes at once — e.g. a chest X-ray showing several conditions simultaneously), `Accuracy`, `Precision`, and `Recall` with `is_multilabel=True` only returned a single averaged scalar. No way to see which specific label the model was struggling with.

Worked example from the PR discussion (`jphdotam`, May 2019):

```
y_true = [(1,1,0), (0,0,0), (1,1,1), (0,1,0)]
y_pred = [(1,0,1), (0,0,1), (0,1,1), (0,1,0)]

per-label accuracy = [0.75, 0.75, 0.5]
```

vs. the current subset-accuracy behavior, which requires an entire row to match and would report something much lower (or, in degenerate cases, 0.0 even when most individual predictions are correct — see reproduction below).

---

## 2. Timeline of everything touching this territory

| Date | Actor | Artifact | Outcome |
|---|---|---|---|
| 2019-05 | jphdotam | Issue #513 opened; PR #516 (`labelwise=True` bool flag on `Accuracy`) | vfdev-5 requested API discussion before merging |
| 2019-05 | vfdev-5 | Review comment on #516 | Rejected bool-flag pattern. Proposed a string-valued `multilabel` arg (`'exact'`/`'labelwise'`/`'top-K'`) replacing `is_multilabel` entirely. Asked to generalize the fix to cover both #513 and #467 (top-K ranking) in one design. |
| 2019-06 | anmolsjoshi | PR #542, continuing #516's approach (still `labelwise=True` bool) | vfdev-5 requested changes: "I'm not a fan of another bool flag." Repeated the string-enum proposal. **PR went silent immediately after this comment and was never updated.** |
| 2019-09 | vfdev-5 | — | Unassigned anmolsjoshi from #513. |
| 2019–2024 | — | — | 5 years of silence on #513 directly. Meanwhile, elsewhere in the codebase, `Precision`/`Recall` were refactored into `_BasePrecisionRecall` with an `average` string param (`False`/`None`/`'micro'`/`'macro'`/`'weighted'`/`'samples'`) — the exact convention now in `master`. This refactor is not mentioned anywhere in #513's own thread; it appears to have happened as separate work and simply left `Accuracy` behind. |
| 2023-03 | julien-blanchon | Issue #467 (top-K precision/recall for ranking) revived, cross-linked to #516 | vfdev-5 reiterates preference for an `average`-style arg over a new bool flag — consistent with his 2019 position. |
| 2025 (~1mo ago) | danijimmy19 | Comment on #513 expressing interest | No PR submitted. |
| 2026-02 | rwtarpit | Issue #3568 opened: generic `TopK` metric wrapper | vfdev-5 endorses the wrapper-class idea (`TopK(Recall(), k=5)`) over per-metric top-K variants. Discussion surfaces that `Accuracy` currently lacks a `_prepare_output` method (unlike `Precision`/`Recall`) and would need one added for the wrapper to work uniformly. |
| 2026-02/27 | rwtarpit | PR #3610 (draft), implementing the wrapper | Three competing state-management designs debated (see §3). Unresolved. |
| 2026-03 | abdelmagid07 | PR #3792: per-metric `TopKMultilabelPrecision`/`TopKMultilabelRecall` classes | Closed by vfdev-5 in favor of the #3568 wrapper approach — "let's go with #3568 approach instead of creating TopK version for each metric." |
| 2026-03 | TahaZahid05 | PR #3627 (fairness metrics) — **merged** | Established a working precedent: a wrapper (`SubgroupAccuracyDifference`) that manages multiple `deepcopy`'d instances of an existing metric internally. Directly relevant precedent for the #3610 design debate (see §3). |
| 2026-06 | star11293 | Comment on #3610 | Suggests scoping the TopK wrapper's first version to multi-label only (binary doesn't apply; multiclass overlaps with existing `TopKCategoricalAccuracy`). Offered to benchmark the two competing state-management approaches. **No reply since.** PR #3610 has had no code activity since ~March and is currently out-of-date with `master`. |

---

## 3. The adjacent effort: TopK wrapper (#3568 / #3610) — why it's adjacent, not blocking

**What it's trying to do:** let any classification metric be wrapped to compute a top-K variant — e.g. `TopK(Recall(), k=5)` for "recall considering only the model's top 5 predictions per sample." Originally motivated by recommender-systems use cases (HitRate@k, MRR, NDCG).

**Why `Accuracy` came up in that discussion at all:** the wrapper needs a uniform hook point across metrics. `Precision`/`Recall` already have `_prepare_output()`; `Accuracy` doesn't. So making the wrapper "just work" on `Accuracy` requires giving `Accuracy` a `_prepare_output` method it currently lacks.

**Why this doesn't block issue #513's fix:**
- Top-K semantics (which predictions count as positive) and label-wise output (whether the result is a scalar or a per-label tensor) are orthogonal axes. Nothing in #513's fix requires or precludes `Accuracy` gaining a `_prepare_output` method later.
- The design for #3610 itself is still unresolved — three different state-management strategies have been proposed and none has consensus:

| Approach | Proposed by | Mechanism | Status |
|---|---|---|---|
| Single instance + state-swap | rwtarpit (original #3610) | One `base_metric` instance; per-k state stored in a dict, swapped via `state_dict()`/`load_state_dict()` | Requires adding a `_skip_checks` flag to base metrics (touches stable API). Distributed sync flagged as broken by reviewer `steaphenai`. |
| Deepcopy per k | TahaZahid05 | `{k: copy.deepcopy(base_metric) for k in ks}` | No changes needed to base metrics. Already shipped in production in PR #3627 (fairness metrics) — the only one of the three with a merged precedent. |
| Transform registry | rwtarpit (evolution of approach 1) | `TopK.register(MetricType, transform_fn)` | vfdev-5 couldn't follow the motivation as of the last comment on it; unresolved. |

- The thread has had **no code changes since March and no comments since June 5**. Waiting for it to resolve before starting on #513 could mean waiting indefinitely.

**Conclusion:** treat #3610 as adjacent context to be aware of and to explicitly not collide with — not a dependency to wait on.

---

## 4. Source-level analysis: why Accuracy's per-label case is NOT a drop-in copy of Precision/Recall's

This is the part that matters most for the actual implementation, and it's not obvious from reading the issue threads alone — it only becomes clear from reading `accuracy.py` and `precision.py` side by side.

### 4.1 How Precision/Recall's per-label output actually works

From `_BasePrecisionRecall._prepare_output` (`precision.py`):

```python
elif self._type == "multilabel":
    num_labels = y_pred.size(1)
    y_pred = torch.transpose(y_pred, 1, -1).reshape(-1, num_labels)
    y = torch.transpose(y, 1, -1).reshape(-1, num_labels)
...
correct = y * y_pred   # elementwise — already naturally "per label"
```

And in `update()`, for `average in [False, None, 'macro', 'weighted']`:

```python
self._denominator += y_pred.sum(dim=0)   # shape (C,) — per-label predicted-positive count
self._numerator += correct.sum(dim=0)    # shape (C,) — per-label true-positive count
```

**The key structural fact:** Precision and Recall's underlying computation (`correct = y * y_pred`) is *already* elementwise/per-label by nature. Getting a per-label result is just a matter of *not collapsing* the label dimension when summing. The per-label numerator and denominator both vary by label (different labels have different numbers of predicted/actual positives), which is exactly why `average='weighted'` and `'micro'` are meaningful — they're different ways of combining label-varying numerators/denominators.

### 4.2 How Accuracy currently works — and why it's different

From `Accuracy.update` (`accuracy.py`):

```python
elif self._type == "multilabel":
    num_classes = y_pred.size(1)
    ...
    correct = torch.all(y == y_pred.type_as(y), dim=-1)   # <-- subset accuracy, collapses labels
```

`torch.all(..., dim=-1)` means **a sample only counts as "correct" if every single label matches.** This is fundamentally different from Precision/Recall's elementwise `correct`. You cannot get a per-label breakdown out of this quantity — the label dimension has already been destroyed by the time `correct` is computed.

**Reproduced locally** (see `reproduce_513.py`): a batch where 9 of 15 individual label predictions were correct (60% element-wise accuracy) still returns `0.0` under current subset-accuracy semantics, because not a single row matched exactly. This is the concrete UX problem the issue is about.

### 4.3 The implication for implementation

Getting label-wise output for `Accuracy` requires computing a **different, elementwise `correct`** — `y == y_pred` **without** the `torch.all(dim=-1)` collapse — not just "changing what we do with the existing `correct` tensor." This is a small code change, but it's a different code path, triggered by the `average` parameter, coexisting with the current subset-accuracy path (which must remain the default for backward compatibility).

### 4.4 A second, more subtle asymmetry: what does "per-label accuracy" even mean, structurally?

For Precision/Recall, the per-label denominator is label-dependent:
- Precision's per-label denominator = predicted-positive count for that label
- Recall's per-label denominator = actual-positive count for that label

For Accuracy, per-label accuracy is:

```
accuracy_label_k = (# samples where y[:,k] == y_pred[:,k]) / N
```

**The denominator is just `N` (total sample count) — identical for every label.** This is structurally simpler than Precision/Recall's per-label case, but it also means the `'weighted'`/`'micro'`/`'samples'` distinctions that make sense for Precision/Recall don't map cleanly onto Accuracy:

- `'macro'` accuracy — mean of per-label accuracies. Clearly meaningful.
- `'micro'` accuracy for multilabel — could mean "overall elementwise accuracy across the flattened (N×C) matrix." This is a real, defined quantity, distinct from macro, and distinct from subset accuracy. Worth supporting.
- `'weighted'` accuracy — weighting per-label accuracy by label positive-count is *possible* but its meaning is less obviously useful than for Precision/Recall (accuracy is already symmetric between positive and negative agreement; weighting by positive support alone is a more arbitrary choice here).
- `'samples'` accuracy — for Precision/Recall this means "compute per-sample, then average over samples." The per-sample analogue for Accuracy — fraction of labels correct for that sample, averaged across samples — is a real, well-known quantity (sometimes called the Hamming score), and is **different from subset accuracy** (which requires *all* labels correct per sample, not just averaging the fraction).

**This asymmetry is exactly the kind of API-design question vfdev-5 has pushed back on twice before (2019, on both #516 and #542).** It should be surfaced explicitly rather than silently deciding it — see the open questions in §6.

### 4.5 The most important asymmetry: what "default" means

For `Precision`/`Recall`, **`average=False` is already the default**, and it already returns a per-label tensor. There is no backward-compatibility problem — the per-label case was the original design.

For `Accuracy`, **the current default (no `average` param, `is_multilabel=True`) returns a single scalar (subset accuracy).** This is load-bearing behavior — every existing user of `Accuracy(is_multilabel=True)` depends on getting a float back. **`average` cannot simply default to `False`/per-label the way it does for Precision/Recall without breaking every existing caller.**

This means the parameter's default value necessarily diverges from the Precision/Recall convention, and that divergence needs to be an explicit, stated design decision — not something that's silently inconsistent with the sibling classes and gets flagged in review.

---

## 5. What this means for scope

Given the above, three possible scopes exist, in increasing order of ambition:

**Scope A — Minimal, additive:**
Add `average` parameter to `Accuracy`, supporting at least `average=False` (per-label tensor via elementwise `correct`, denominator = N per label). Default behavior (no `average` passed) stays exactly as-is for backward compatibility. Does not touch `_prepare_output`, does not interact with #3610.

**Scope B — Full parity:**
Same as A, but supports the complete `average` vocabulary (`'micro'`, `'macro'`, `'weighted'`, `'samples'`) with the semantics worked out in §4.4, each carefully distinguished from subset accuracy.

**Scope C — Structural unification:**
Also add `_prepare_output` to `Accuracy` so it can participate in the #3610 TopK wrapper machinery, effectively contributing to #3568 as a side effect. Higher risk of scope creep and direct entanglement with a currently-unresolved design debate.

**Recommendation: start with Scope A, explicitly framed to maintainer as extensible to B, and out of scope for C unless requested.** This is proposed, not decided — see open questions below.

---

## 6. Open questions for the maintainer (to resolve before implementation)

1. Is a narrowly-scoped `average` parameter (Scope A) acceptable as a first PR, or should this wait for #3610 to resolve so `Accuracy` can be refactored in one pass?
2. Should the initial implementation support just `average=False`, or the fuller vocabulary (Scope B)? If the latter, what should `'weighted'`, `'micro'`, and `'samples'` mean for Accuracy specifically, given they don't map onto the Precision/Recall semantics unchanged (§4.4)?
3. Given `Accuracy`'s current default cannot become per-label without breaking backward compatibility (§4.5), is `average=None` (preserving current scalar/subset behavior) an acceptable default, with `average=False` as the explicit opt-in for per-label output?
4. Any preference on `_state_dict_all_req_keys` handling once `_num_correct` can be either a scalar or a tensor depending on `average`?

---

## 7. Reproduction reference

Script: `reproduce_513.py` in this repo. Run against ignite v0.6.0 on `master`.

| Metric | Call | Output | Status |
|---|---|---|---|
| `Recall(is_multilabel=True)` | default | per-label tensor | Already fixed |
| `Precision(is_multilabel=True)` | default | per-label tensor | Already fixed |
| `Accuracy(is_multilabel=True)` | default | scalar float (subset accuracy) | **Gap — this PR's target** |

---

## 8. Files expected to change

- `ignite/metrics/accuracy.py` — `Accuracy.__init__`, `.reset`, `.update`, `.compute`; possibly `_state_dict_all_req_keys`
- `tests/ignite/metrics/test_accuracy.py` — new per-label test cases, modeled on the multilabel + `average` tests already in `tests/ignite/metrics/test_precision.py`
- `docs/source/metrics.rst` — usage examples, following the `Precision`/`Recall` docstring pattern already in `precision.py`
