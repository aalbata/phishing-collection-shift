"""Metrics for every evaluation population, computed only from saved scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import at_threshold, projected_precision, ranking

# Population name -> scored frames that form it (positives from the first, negatives from the second).
POPULATIONS = {
    "in_distribution": ("P_test", "L_test"),
    "phishing_shift": ("Pother_all", "L_test"),
    "legitimate_shift": ("P_test", "Lother_all"),
    "both_shift": ("Pother_all", "Lother_all"),
    "modern_legitimate": ("P_test", "Mod_all"),
    "phishing_shift_modern_legitimate": ("Pother_all", "Mod_all"),
}
ASSUMED_PREVALENCE = (0.001, 0.01)


def run_dirs(root: Path):
    for done in sorted(Path(root).glob("*/rep*/*__*/done.json")):
        yield done.parent


def population(scores: pd.DataFrame, name: str) -> pd.DataFrame:
    pos_frame, neg_frame = POPULATIONS[name]
    pos = scores[(scores.frame == pos_frame) & (scores.label == 1)]
    neg = scores[(scores.frame == neg_frame) & (scores.label == 0)]
    return pd.concat([pos, neg], ignore_index=True)


def run_metrics(d: Path) -> list[dict]:
    sel = json.loads((d / "selection.json").read_text(encoding="utf-8"))
    thr = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
    scores = pd.read_parquet(d / "scores.parquet")
    rows = []
    key = dict(pairing=sel["pairing"], replicate=sel["replicate"], variant=sel["variant"],
               weighting=sel["weighting"], chosen_tune_ap=sel["chosen_tune_ap"])
    for pop in POPULATIONS:
        z = population(scores, pop)
        if z.label.nunique() < 2:
            continue  # population not formed (a collection is absent)
        y, s = z.label.to_numpy(), z.score.to_numpy()
        r = ranking(y, s)
        r["brier"] = float(np.mean((s - y) ** 2))
        rows.append(dict(key, population=pop, alpha=None, **r))
        for t in thr:
            row = dict(key, population=pop, alpha=t["alpha"], threshold_status=t["status"],
                       n_calibration_negative=t["n_calibration_negative"], threshold=t["threshold"],
                       n=r["n"], n_positive=r["n_positive"], n_negative=r["n_negative"], prevalence=r["prevalence"])
            if t["threshold"] is not None:
                m = at_threshold(y, s, t["threshold"])
                row.update(m)
                row["fpr_within_alpha"] = bool(m["fpr"] <= t["alpha"])
                for pi in ASSUMED_PREVALENCE:
                    row[f"projected_precision_at_{pi}"] = projected_precision(m["recall"], m["fpr"], pi)
            rows.append(row)
    return rows


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for d in run_dirs(root):
        rows.extend(run_metrics(d))
    df = pd.DataFrame(rows)
    # Selected weighting per (pairing, replicate, variant): larger tuning AP, "none" on ties.
    tune = (df[df.alpha.isna() & df.population.eq("in_distribution")]
            .pivot_table(index=["pairing", "replicate", "variant"], columns="weighting",
                         values="chosen_tune_ap").reset_index())
    tune["selected_weighting"] = np.where(tune["balanced"] > tune["none"], "balanced", "none")
    df = df.merge(tune[["pairing", "replicate", "variant", "selected_weighting"]],
                  on=["pairing", "replicate", "variant"], how="left")
    df["selected"] = df.weighting.eq(df.selected_weighting)
    return df


def summarise(df: pd.DataFrame, value: str, alpha=None) -> pd.DataFrame:
    z = df[df.selected & (df.alpha.isna() if alpha is None else df.alpha.eq(alpha))]
    g = z.groupby(["pairing", "variant", "population"])[value]
    out = g.agg(["mean", "std", "min", "max", "count"]).reset_index()
    return out
