"""Numerical metrics used by recording figure scripts."""

from __future__ import annotations

import math

import numpy as np


def normalized_distribution(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return a finite probability vector with the same shape as `values`."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a rank-1 vector, got shape {arr.shape}")
    if np.any(arr < 0):
        raise ValueError("probability vectors must be non-negative")
    total = float(arr.sum())
    if not np.isfinite(total):
        raise ValueError("probability vector contains non-finite values")
    if total <= eps:
        return np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=np.float64)
    return arr / total


def js_divergence(left: np.ndarray, right: np.ndarray, *, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in bits for two probability vectors."""
    p = normalized_distribution(left, eps=eps)
    q = normalized_distribution(right, eps=eps)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def pairwise_js(distributions: np.ndarray) -> np.ndarray:
    """Compute an NxN Jensen-Shannon distance matrix."""
    if distributions.ndim != 2:
        raise ValueError(f"expected rank-2 matrix, got {distributions.shape}")
    probs = _normalize_rows(distributions)
    log_probs = np.log2(np.clip(probs, 1e-12, 1.0))
    entropy = -np.sum(probs * log_probs, axis=1)

    mixture = 0.5 * (probs[:, None, :] + probs[None, :, :])
    log_mixture = np.log2(np.clip(mixture, 1e-12, 1.0))
    mixture_entropy = -np.sum(mixture * log_mixture, axis=2)
    distances = mixture_entropy - 0.5 * (entropy[:, None] + entropy[None, :])
    return np.maximum(distances, 0.0)


def _normalize_rows(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr < 0) or np.any(~np.isfinite(arr)):
        raise ValueError("distribution matrix contains negative or non-finite values")
    totals = arr.sum(axis=1, keepdims=True)
    out = np.divide(arr, totals, out=np.zeros_like(arr), where=totals > eps)
    empty = (totals[:, 0] <= eps)
    if bool(empty.any()):
        out[empty] = 1.0 / max(arr.shape[1], 1)
    return out


def specialization_score(distribution: np.ndarray) -> float:
    """Return 1 - normalized entropy; higher means more concentrated."""
    p = normalized_distribution(distribution)
    active = p[p > 0]
    if active.size <= 1:
        return 1.0
    entropy = float(-np.sum(active * np.log(active)))
    max_entropy = math.log(float(p.size))
    if max_entropy <= 0:
        return 1.0
    return 1.0 - entropy / max_entropy


def pearson_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Pearson correlation, or NaN when undefined."""
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
