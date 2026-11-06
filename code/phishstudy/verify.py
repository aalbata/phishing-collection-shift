"""Independent recomputation of every reported number from saved scores.

This module deliberately does not import the study's metric, threshold or
analysis code. It re-implements average precision, ROC-AUC (Mann-Whitney with
mid-ranks), confusion counts, the Neyman-Pearson order-statistic rank (exact
binomial sums in log space) and Clopper-Pearson limits (bisection on exact
binomial sums), then compares them with the stored values. Any mismatch raises
``AuditError``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9
POPULATIONS = {
    "in_distribution": ("P_test", "L_test"),
    "phishing_shift": ("Pother_all", "L_test"),
    "legitimate_shift": ("P_test", "Lother_all"),
    "both_shift": ("Pother_all", "Lother_all"),
    "modern_legitimate": ("P_test", "Mod_all"),
    "phishing_shift_modern_legitimate": ("Pother_all", "Mod_all"),
}


class AuditError(AssertionError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AuditError(msg)


def ap(y, s) -> float:
    """Average precision: sum over distinct thresholds of (R_k - R_{k-1}) * P_k."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[last_of_tie]
    fp = (last_of_tie + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def auc(y, s) -> float:
    """ROC-AUC as the Mann-Whitney statistic with mid-ranks for ties."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    starts = np.r_[0, np.flatnonzero(np.diff(ss) != 0) + 1]
    ends = np.r_[starts[1:], len(ss)]
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a + b - 1) / 2 + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _log_pmf(k: int, n: int, p: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _logsum(terms) -> float:
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def upper_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), summing whichever side has fewer terms."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    if n - k + 1 <= k:
        return math.exp(_logsum([_log_pmf(j, n, p) for j in range(k, n + 1)]))
    return max(0.0, 1.0 - math.exp(_logsum([_log_pmf(j, n, p) for j in range(0, k)])))


def cdf(k: int, n: int, p: float) -> float:
    """P(X <= k)."""
    return 1.0 - upper_tail(k + 1, n, p)


def np_rank(n: int, alpha: float, delta: float):
    need = math.ceil(math.log(delta) / math.log(1 - alpha))
    if n < need:
        return None, need
    k = max(1, int(n * (1 - alpha)) - 6 * int(math.sqrt(n) + 1))
    while upper_tail(k, n, 1 - alpha) > delta:
        k += 1
    while k > 1 and upper_tail(k - 1, n, 1 - alpha) <= delta:
        k -= 1
    return k, need


def _bisect(f, increasing: bool, target: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _tail_beta(k: int, n: int, p: float) -> float:
    """P(X >= k) = I_p(k, n - k + 1), the regularized incomplete beta function (k >= 1)."""
    from scipy.special import betainc
    return float(betainc(k, n - k + 1, p))


def clopper_pearson(k: int, n: int, level: float = 0.95):
    """Clopper-Pearson limits by bisection on the binomial tail (analysis code uses beta quantiles instead)."""
    if n == 0:
        return [None, None]
    a = (1 - level) / 2
    lower = 0.0 if k == 0 else _bisect(lambda p: _tail_beta(k, n, p), True, a)
    upper = 1.0 if k == n else _bisect(lambda p: 1.0 - _tail_beta(k + 1, n, p), False, a)
    return [lower, upper]


def close(a, b, tol=TOL) -> bool:
    def missing(v):
        return v is None or (isinstance(v, float) and math.isnan(v))
    if missing(a) or missing(b):
        return missing(a) and missing(b)
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def confusion(y, s, t):
    pred = np.asarray(s) > t
    y = np.asarray(y) == 1
    return dict(tp=int((pred & y).sum()), fp=int((pred & ~y).sum()), tn=int((~pred & ~y).sum()),
                fn=int((~pred & y).sum()))


def audit_runs(runs_root: Path, units: pd.DataFrame, splits: pd.DataFrame, pairings: dict, other: dict,
               metrics: pd.DataFrame, ci_tol: float = 1e-6) -> dict:
    """Check scores, memberships, thresholds and every stored metric row."""
    label = units.set_index("record_id")["label"]
    coll = units.set_index("record_id")["collection"]
    checked_rows = 0
    runs = 0
    for done in sorted(Path(runs_root).glob("*/rep*/*__*/done.json")):
        d = done.parent
        sel = json.loads((d / "selection.json").read_text(encoding="utf-8"))
        thr = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
        sc = pd.read_parquet(d / "scores.parquet")
        pairing, rep = sel["pairing"], sel["replicate"]
        p, l = pairings[pairing]
        sp = splits[splits.replicate.eq(rep)].set_index("record_id")["split"]
        check(np.isfinite(sc.score).all(), f"non-finite score in {d}")
        check(not sc.duplicated(["frame", "record_id"]).any(), f"duplicate scored unit in {d}")
        check(not sc.record_id.duplicated().any(), f"a unit is scored in two frames in {d}")
        check((sc.label.to_numpy() == label.loc[sc.record_id].to_numpy()).all(), f"label mismatch in {d}")
        check((sc.collection.to_numpy() == coll.loc[sc.record_id].to_numpy()).all(), f"collection mismatch {d}")
        expect = {
            "P_calibration": set(units.record_id[units.collection.eq(p) & units.record_id.map(sp).eq("calibration")]),
            "L_calibration": set(units.record_id[units.collection.eq(l) & units.record_id.map(sp).eq("calibration")]),
            "P_test": set(units.record_id[units.collection.eq(p) & units.record_id.map(sp).eq("test")]),
            "L_test": set(units.record_id[units.collection.eq(l) & units.record_id.map(sp).eq("test")]),
            "Pother_all": set(units.record_id[units.collection.eq(other[p])]),
            "Lother_all": set(units.record_id[units.collection.eq(other[l])]),
            "Mod_all": set(units.record_id[units.collection.eq("apache_lists")]),
        }
        for frame, ids in expect.items():
            check(set(sc.record_id[sc.frame.eq(frame)]) == ids, f"{frame} membership differs from splits in {d}")
        trained = set(units.record_id[units.collection.isin([p, l]) & units.record_id.map(sp).isin(["fit", "tune"])])
        check(not (trained & set(sc.record_id)), f"fit/tune units were scored as evaluation units in {d}")
        neg = np.sort(sc[(sc.frame == "L_calibration") & (sc.label == 0)].score.to_numpy())
        for t in thr:
            k, need = np_rank(len(neg), t["alpha"], t["delta"])
            check(t["required"] == need and t["n_calibration_negative"] == len(neg), f"calibration count {d}")
            if k is None:
                check(t["threshold"] is None, f"threshold reported without enough negatives in {d}")
            else:
                check(t["rank"] == k and t["threshold"] == float(neg[k - 1]), f"threshold not reproduced in {d}")
        m = metrics[(metrics.pairing == pairing) & (metrics.replicate == rep) & (metrics.variant == sel["variant"])
                    & (metrics.weighting == sel["weighting"])]
        formed = [pop for pop, (pf, nf) in POPULATIONS.items()
                  if ((sc.frame == pf) & (sc.label == 1)).any() and ((sc.frame == nf) & (sc.label == 0)).any()]
        check(len(m) == len(formed) * (1 + len(thr)), f"unexpected metric row count for {d}")
        for pop in formed:
            pf, nf = POPULATIONS[pop]
            z = pd.concat([sc[(sc.frame == pf) & (sc.label == 1)], sc[(sc.frame == nf) & (sc.label == 0)]])
            y, s = z.label.to_numpy(), z.score.to_numpy()
            row = m[(m.population == pop) & m.alpha.isna()].iloc[0]
            check(row.n == len(y) and row.n_positive == int(y.sum()), f"counts {pop} {d}")
            check(close(row.average_precision, ap(y, s)), f"AP mismatch {pop} {d}")
            check(close(row.roc_auc, auc(y, s)), f"ROC-AUC mismatch {pop} {d}")
            check(close(row.brier, float(sum((si - yi) ** 2 for si, yi in zip(s.tolist(), y.tolist())) / len(y))),
                  f"Brier score mismatch {pop} {d}")
            checked_rows += 1
            for t in thr:
                r = m[(m.population == pop) & m.alpha.eq(t["alpha"])].iloc[0]
                if t["threshold"] is None:
                    check(pd.isna(r.get("tp")), f"operating point reported for an unavailable threshold {d}")
                    continue
                c = confusion(y, s, t["threshold"])
                for key, v in c.items():
                    check(int(r[key]) == v, f"{key} mismatch {pop} alpha={t['alpha']} {d}")
                pos, negs = c["tp"] + c["fn"], c["tn"] + c["fp"]
                check(close(r.recall, c["tp"] / pos), f"recall {d}")
                check(close(r.fpr, c["fp"] / negs), f"fpr {d}")
                prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
                check(close(r.precision, prec), f"precision {d}")
                tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
                rec, fpr_ = tp / pos, fp / negs
                check(close(r.specificity, tn / negs), f"specificity {d}")
                check(close(r.f1, 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None), f"F1 {d}")
                mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
                check(close(r.mcc, (tp * tn - fp * fn) / mcc_den if mcc_den else None), f"MCC {d}")
                check(close(r.balanced_accuracy, 0.5 * (rec + tn / negs)), f"balanced accuracy {d}")
                for pi in (0.001, 0.01):
                    den = pi * rec + (1 - pi) * fpr_
                    check(close(r[f"projected_precision_at_{pi}"], pi * rec / den if den > 0 else None),
                          f"projected precision at {pi} {d}")
                lo, hi = clopper_pearson(c["fp"], negs)
                check(abs(r.fpr_ci[0] - lo) < ci_tol and abs(r.fpr_ci[1] - hi) < ci_tol, f"FPR interval {d}")
                lo, hi = clopper_pearson(c["tp"], pos)
                check(abs(r.recall_ci[0] - lo) < ci_tol and abs(r.recall_ci[1] - hi) < ci_tol, f"recall CI {d}")
                checked_rows += 1
        runs += 1
    check(runs > 0, "no runs found")
    return dict(runs=runs, metric_rows_checked=checked_rows)
