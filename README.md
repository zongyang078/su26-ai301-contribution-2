# Contribution 2: [Issue Title]

**Contribution Number:** [ 2 ]  
**Student:** Zongyang Li
**Issue:** https://github.com/pytorch/ignite/issues/513  
**Status:** [Phase I ] [In Progress]

---

## Why I Chose This Issue

I chose this issue because it lives in the same ignite/metrics/ module as my first contribution (issue #1757, EpochMetric output types), so I can build directly on the environment, testing patterns, and code style I already learned. It's also a natural extension of that work: multi-label metric averaging is a well-scoped, well-understood problem, and the maintainers (vfdev-5) already sketched an implementation direction back in 2019 via PR #516, which gives me a concrete reference point rather than an open-ended design problem.

Technically, this matches my interest in evaluation/metrics engineering for ML systems — I want to get comfortable extending statistics/accumulator-style classes (reset/update/compute lifecycle) and handling tensor-shape edge cases correctly, which is directly relevant to the AI/ML engineering roles I'm applying for.

---

## Understanding the Issue

### Problem Description

In multi-label classification (each sample can belong to multiple, non-mutually-exclusive classes, e.g. y = [0, 1, 1]), Ignite's Accuracy, Precision, and Recall metrics currently only support averaged multi-label scores — a single scalar summarizing performance across all labels. There is no way to get per-label (label-wise) scores.

### Expected Behavior

A user should be able to request per-label metric values, e.g. 3 separate accuracy scores for a 3-label problem, so they can identify which specific label is driving overall error.

### Current Behavior

- Accuracy(is_multilabel=True) always averages across labels/samples into one number; there's no flag to disable this.
- Precision(is_multilabel=True, average=False) does not return per-label scores either — it returns a per-sample binary result instead of a per-label score, which is not what users expect.

### Affected Components

ignite/metrics/accuracy.py (Accuracy, and shared logic in _BaseClassification)
ignite/metrics/precision.py (Precision)
ignite/metrics/recall.py (Recall)
Corresponding test files under tests/ignite/metrics/
Possibly shared logic in ignite/metrics/metrics_lambda.py if label-wise output needs special handling downstream

---

## Reproduction Process

### Environment Setup

Reusing the ignite-dev conda environment (Python 3.10) set up for my first contribution — no changes needed. Will re-sync my existing fork (zongyang078/ignite) with upstream/main before starting, since it's been several months since my last contribution and the codebase has moved from the 2019-era metrics implementation referenced in the original issue to the current v0.5.x structure (confirmed via current source: ignite/metrics/accuracy.py now includes a shared _BaseClassification base class and additional skip_unrolling support not present in 2019).

### Steps to Reproduce

1. Construct a multi-label batch, e.g. y_true and y_pred of shape (batch_size, num_labels) with 0/1 values.
2. Attach Accuracy(is_multilabel=True) and run — observe only a single averaged scalar is returned.
3. Attempt Precision(is_multilabel=True, average=False) — observe the returned tensor has shape (batch_size,) (per-sample), not (num_labels,) (per-label) as a user might expect.

### Reproduction Evidence

- **Commit showing reproduction:** [Link to commit in your fork]
- **Screenshots/logs:** [If applicable]
- **My findings:** [What you discovered during reproduction]

---

## Solution Approach

### Analysis

[Your analysis of the root cause - what's causing the issue?]

### Proposed Solution

[High-level description of your fix approach]

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** [Restate the problem]

**Match:** [What similar patterns/solutions exist in the codebase?]

**Plan:** [Step-by-step implementation plan]
1. [Modify file X to do Y]
2. [Add function Z]
3. [Update tests]

**Implement:** [Link to your branch/commits as you work]

**Review:** [Self-review checklist - does it follow the project's contribution guidelines?]

**Evaluate:** [How will you verify it works?]

---

## Testing Strategy

### Unit Tests

- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Test case 3: [Description]

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

[What you tested manually and results]

---

## Implementation Notes

### Week [X] Progress

[What you built this week, challenges faced, decisions made]

### Week [Y] Progress

[Continue documenting as you work]

### Code Changes

- **Files modified:** [List]
- **Key commits:** [Links to important commits]
- **Approach decisions:** [Why you chose certain approaches]

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
