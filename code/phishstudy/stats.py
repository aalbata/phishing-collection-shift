"""Thresholds, metrics and resampling inference computed from saved scores."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, binom
from sklearn.metrics import average_precision_score, roc_auc_score


def minimum_negatives(alpha: float, delta: float = 0.05) -> int:
    """Smallest benign calibration count for which the order-statistic rule exists."""
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError("alpha and delta must lie in (0, 1)")
    return int(np.ceil(np.log(delta) / np.log1p(-alpha)))


def np_threshold(negative_scores, alpha: float, delta: float = 0.05) -> dict:
    """Neyman-Pearson umbrella threshold (Tong, Feng and Li, 2018).

    With n benign calibration scores sorted increasingly, choose the k-th order
    statistic with the smallest k such that P(Binomial(n, 1 - alpha) >= k) <= delta.
    A message is flagged when its score is strictly greater than the threshold.
    Then P(population FPR > alpha) <= delta for exchangeable benign messages from
    the calibration distribution; nothing is implied for a different collection.
    """
    x = np.sort(np.asarray(negative_scores, dtype=float))
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("invalid calibration scores")
    n, need = len(x), minimum_negatives(alpha, delta)
    base = dict(alpha=alpha, delta=delta, n_calibration_negative=n, required=need)
    if n < need:
        return dict(base, threshold=None, rank=None, violation_bound=None, status="UNAVAILABLE_TOO_FEW_NEGATIVES")
    ranks = np.arange(1, n + 1)
    tails = binom.sf(ranks - 1, n, 1 - alpha)
    k = int(ranks[np.flatnonzero(tails <= delta)[0]])
    return dict(base, threshold=float(x[k - 1]), rank=k, violation_bound=float(tails[k - 1]), status="CALIBRATED")


def clopper_pearson(k: int, n: int, level: float = 0.95):
    if n == 0:
        return [None, None]
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [lo, hi]


def ranking(y, s) -> dict:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if len(y) != len(s) or not np.isfinite(s).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("ranking metrics need aligned finite scores and both classes")
    return dict(n=int(len(y)), n_positive=int(y.sum()), n_negative=int((1 - y).sum()), prevalence=float(y.mean()),
                average_precision=float(average_precision_score(y, s)), roc_auc=float(roc_auc_score(y, s)))


def at_threshold(y, s, t: float) -> dict:
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(s, dtype=float) > t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pos, neg = tp + fn, tn + fp
    recall = tp / pos if pos else None
    fpr = fp / neg if neg else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / denom) if denom else None
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, recall=recall, fpr=fpr, specificity=None if fpr is None else 1 - fpr,
                precision=precision, f1=f1, mcc=mcc,
                balanced_accuracy=None if recall is None or fpr is None else (recall + 1 - fpr) / 2,
                recall_ci=clopper_pearson(tp, pos), fpr_ci=clopper_pearson(fp, neg))


def projected_precision(recall: float, fpr: float, prevalence: float):
    """Precision implied by fixed class-conditional rates at an assumed prevalence (algebra only)."""
    den = prevalence * recall + (1 - prevalence) * fpr
    return prevalence * recall / den if den > 0 else None


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, p[i] * (len(p) - rank)))
        adj[i] = running
    return adj


def _ap_weighted(y, s, w):
    return average_precision_score(y, s, sample_weight=w)


def paired_bootstrap(replicates, metric="average_precision", n_boot: int = 2000, seed: int = 20261105):
    """Two-level paired bootstrap of the difference between two models.

    ``replicates`` is a list of (y, score_a, score_b) arrays, one per fitting
    replicate, aligned on the same test records. Each bootstrap draw resamples
    replicates with replacement and then, inside each drawn replicate, resamples
    positive and negative test records with replacement (stratified). The
    statistic is the mean over drawn replicates of metric(a) - metric(b). Only
    the multiplicities of observed records change; no message is created.
    """
    rng = np.random.default_rng(seed)
    observed = []
    for y, a, b in replicates:
        if metric == "average_precision":
            observed.append(average_precision_score(y, a) - average_precision_score(y, b))
        else:
            raise ValueError(metric)
    observed = float(np.mean(observed))
    draws = np.empty(n_boot)
    R = len(replicates)
    idx = [(np.flatnonzero(np.asarray(y) == 1), np.flatnonzero(np.asarray(y) == 0)) for y, _, _ in replicates]
    for bi in range(n_boot):
        vals = []
        for r in rng.integers(0, R, size=R):
            y, a, b = replicates[r]
            pos, neg = idx[r]
            w = np.zeros(len(y))
            np.add.at(w, rng.choice(pos, size=len(pos), replace=True), 1)
            np.add.at(w, rng.choice(neg, size=len(neg), replace=True), 1)
            vals.append(_ap_weighted(y, a, w) - _ap_weighted(y, b, w))
        draws[bi] = np.mean(vals)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return dict(delta=observed, ci_low=float(lo), ci_high=float(hi), p_value=float(min(1.0, max(p, 1 / n_boot))),
                n_boot=n_boot, replicates=R, seed=seed)
