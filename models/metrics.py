"""models.metrics — evaluation metrics for the BFId baseline classifier.

The paper (Todt, Morsbach, Strufe, CCS '25) reports:
  * top-1 identity accuracy (headline 99.5% +- 0.38 for BFI / normal walking),
  * top-2 accuracy for the empty-room control (sec. 5.3),
  * accuracy broken down *by participant*, *by style* and *by perspective*,
  * mean +- std over repeated (seeded) train/test splits.

All helpers operate on plain numpy arrays so they have no torch dependency and
are trivially unit-testable. Labels are the integer class indices produced by
:class:`models.dataset.BFIdDataset` (use its ``idx_to_label`` to map back to
participant ids for reporting).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import mean, pstdev

import numpy as np

from wallflower import contract  # noqa: F401  (imported so naming stays anchored)


def _as_array(a: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(a)


def accuracy(y_true: Sequence[int] | np.ndarray,
             y_pred: Sequence[int] | np.ndarray) -> float:
    """Top-1 accuracy. Returns 0.0 for an empty input."""
    yt, yp = _as_array(y_true), _as_array(y_pred)
    if yt.size == 0:
        return 0.0
    return float((yt == yp).mean())


def top_k_accuracy(y_true: Sequence[int] | np.ndarray,
                   logits: np.ndarray, k: int = 2) -> float:
    """Top-k accuracy from a ``[N, num_classes]`` logit/score matrix.

    The paper uses top-2 for the empty-room control (sec. 5.3): a hit counts if
    the true id is among the ``k`` highest-scoring classes.
    """
    yt = _as_array(y_true)
    scores = np.asarray(logits)
    if yt.size == 0:
        return 0.0
    k = min(k, scores.shape[1])
    # indices of the top-k scores per row (unordered top-k is fine for membership)
    topk = np.argpartition(scores, -k, axis=1)[:, -k:]
    hits = (topk == yt[:, None]).any(axis=1)
    return float(hits.mean())


def per_group_accuracy(y_true: Sequence[int] | np.ndarray,
                       y_pred: Sequence[int] | np.ndarray,
                       groups: Sequence) -> dict:
    """Accuracy within each group key (e.g. style, perspective, participant).

    Returns ``{group_key: {"accuracy": float, "n": int}}`` sorted by key.
    """
    yt, yp = _as_array(y_true), _as_array(y_pred)
    buckets: dict[object, list[bool]] = defaultdict(list)
    for t, p, g in zip(yt.tolist(), yp.tolist(), groups, strict=True):
        buckets[g].append(t == p)
    out: dict = {}
    for g in sorted(buckets, key=lambda x: (str(type(x)), x)):
        hits = buckets[g]
        out[g] = {"accuracy": float(mean(hits)), "n": len(hits)}
    return out


def confusion_matrix(y_true: Sequence[int] | np.ndarray,
                     y_pred: Sequence[int] | np.ndarray,
                     num_classes: int) -> np.ndarray:
    """Dense ``[num_classes, num_classes]`` confusion matrix (rows=true)."""
    yt, yp = _as_array(y_true), _as_array(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(yt.tolist(), yp.tolist(), strict=True):
        cm[t, p] += 1
    return cm


def mean_std(values: Sequence[float]) -> dict:
    """Mean +- (population) std over repeated splits, paper-style reporting."""
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(mean(vals)),
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def format_mean_std(values: Sequence[float], scale: float = 100.0,
                    unit: str = "%") -> str:
    """Render ``mean +- std`` like the paper's '99.5% +- 0.38'."""
    ms = mean_std(values)
    return f"{ms['mean'] * scale:.2f}{unit} +- {ms['std'] * scale:.2f}"
