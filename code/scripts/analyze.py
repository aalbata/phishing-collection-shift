"""Compute all metrics, summaries and declared tests from saved scores.

Outputs go to evidence/metrics/. Every table and figure of the manuscript is
built from these files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from phishstudy.runtime import disable_power_throttling  # noqa: E402

disable_power_throttling()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from phishstudy.analysis import POPULATIONS, collect, population  # noqa: E402
from phishstudy.stats import at_threshold, holm, paired_bootstrap  # noqa: E402

PRIMARY = "NZ-SA"
CONFIRMATORY = [("feature_token_transformer", "mlp"), ("feature_token_transformer", "single_token_attention"),
                ("feature_token_transformer", "gradient_boosting"), ("feature_token_transformer", "tfidf_logistic")]
CONFIRMATORY_POPULATIONS = ("in_distribution", "both_shift")


def load_scores(runs: Path, pairing, rep, variant, weighting) -> pd.DataFrame:
    return pd.read_parquet(runs / pairing / f"rep{rep}" / f"{variant}__{weighting}" / "scores.parquet")


def confirmatory(runs: Path, metrics: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    sel = metrics[metrics.alpha.isna() & metrics.population.eq("in_distribution")]
    chosen = (sel.drop_duplicates(["pairing", "replicate", "variant"])
              .set_index(["pairing", "replicate", "variant"])["selected_weighting"])
    rows = []
    for pop in CONFIRMATORY_POPULATIONS:
        for a, b in CONFIRMATORY:
            reps = []
            for rep in sorted(metrics[metrics.pairing.eq(PRIMARY)].replicate.unique()):
                za = population(load_scores(runs, PRIMARY, rep, a, chosen[(PRIMARY, rep, a)]), pop)
                zb = population(load_scores(runs, PRIMARY, rep, b, chosen[(PRIMARY, rep, b)]), pop)
                m = za.merge(zb, on=["record_id", "label"], suffixes=("_a", "_b"), validate="one_to_one")
                if len(m) != len(za) or len(m) != len(zb):
                    raise AssertionError("paired populations differ")
                m = m.sort_values("record_id")
                reps.append((m.label.to_numpy(), m.score_a.to_numpy(), m.score_b.to_numpy()))
            r = paired_bootstrap(reps, n_boot=n_boot)
            rows.append(dict(pairing=PRIMARY, population=pop, model_a=a, model_b=b, **r))
    df = pd.DataFrame(rows)
    df["holm_p"] = holm(df.p_value.to_numpy())
    df["reject_at_0.05"] = df.holm_p < 0.05
    return df


def low_prevalence(runs: Path, metrics: pd.DataFrame, prevalence=0.01, draws=100, seed=20261105) -> pd.DataFrame:
    """Subsets of real positives drawn without replacement so that positives form 1% of the population.

    As declared in the protocol, every population with at least 10,000 legitimate units qualifies; in this corpus these
    are the populations whose legitimate units are all Enron units or all Apache list units.
    """
    rng = np.random.default_rng(seed)
    rows = []
    sel = metrics[metrics.selected & metrics.alpha.eq(0.01)]
    for _, r in sel.iterrows():
        if r.n_negative < 10000 or pd.isna(r.threshold):
            continue
        z = population(load_scores(runs, r.pairing, r.replicate, r.variant, r.weighting), r.population)
        pos, neg = z[z.label.eq(1)], z[z.label.eq(0)]
        k = int(round(prevalence / (1 - prevalence) * len(neg)))
        if k > len(pos):
            continue
        aps, precs = [], []
        from sklearn.metrics import average_precision_score
        for _ in range(draws):
            sub = pd.concat([pos.iloc[rng.choice(len(pos), size=k, replace=False)], neg])
            aps.append(average_precision_score(sub.label, sub.score))
            m = at_threshold(sub.label.to_numpy(), sub.score.to_numpy(), r.threshold)
            precs.append(np.nan if m["precision"] is None else m["precision"])
        rows.append(dict(pairing=r.pairing, replicate=r.replicate, variant=r.variant, population=r.population,
                         n_negative=len(neg), n_positive_subset=k, draws=draws,
                         ap_mean=float(np.mean(aps)), ap_q025=float(np.quantile(aps, .025)),
                         ap_q975=float(np.quantile(aps, .975)), precision_mean=float(np.nanmean(precs)),
                         precision_q025=float(np.nanquantile(precs, .025)),
                         precision_q975=float(np.nanquantile(precs, .975))))
    return pd.DataFrame(rows)


def importance_tables(runs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp, att = [], []
    for f in sorted(runs.glob(f"{PRIMARY}/rep*/*__*/permutation_importance.json")):
        variant, weighting = f.parent.name.split("__")
        rep = int(f.parent.parent.name[3:])
        d = json.loads(f.read_text(encoding="utf-8"))
        for pop in ("in_distribution", "both_shift"):
            for feat, v in d[pop]["importance"].items():
                imp.append(dict(replicate=rep, variant=variant, weighting=weighting, population=pop, feature=feat,
                                mean_drop=v["mean"], sd_drop=v["sd"], base_ap=d[pop]["base_average_precision"]))
        a = f.parent / "cls_attention.json"
        if a.exists():
            w = json.loads(a.read_text(encoding="utf-8"))["weights"]
            for feat, v in w.items():
                att.append(dict(replicate=rep, weighting=weighting, feature=feat, attention=v))
    return pd.DataFrame(imp), pd.DataFrame(att)


def error_profile(runs: Path, metrics: pd.DataFrame, units: pd.DataFrame,
                  variants=("feature_token_transformer", "tfidf_logistic", "gradient_boosting")) -> pd.DataFrame:
    """Aggregate miss and false-alarm rates by message properties at the alpha = 0.01 threshold (no text)."""
    u = units.set_index("record_id")
    rows = []
    sel = metrics[metrics.selected & metrics.alpha.eq(0.01) & metrics.pairing.eq(PRIMARY)
                  & metrics.variant.isin(variants) & metrics.population.isin(["in_distribution", "both_shift"])]
    for _, r in sel.iterrows():
        if pd.isna(r.threshold):
            continue
        z = population(load_scores(runs, r.pairing, r.replicate, r.variant, r.weighting), r.population)
        z = z.join(u[["html_only", "url_count", "words", "attachments", "year_header"]], on="record_id")
        z["flagged"] = z.score > r.threshold
        z["length"] = pd.cut(z.words, [-1, 49, 199, np.inf], labels=["<50 words", "50-199 words", ">=200 words"])
        z["links"] = np.where(z.url_count > 0, "has links", "no links")
        z["format"] = np.where(z.html_only == 1, "HTML only", "plain text present")
        for prop in ("length", "links", "format"):
            for (label, level), g in z.groupby(["label", prop], observed=True):
                rows.append(dict(replicate=r.replicate, variant=r.variant, population=r.population,
                                 property=prop, level=str(level), label=int(label), n=len(g),
                                 flagged=int(g.flagged.sum()),
                                 rate=float(g.flagged.mean())))
    return pd.DataFrame(rows)


def compute_table(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/rep*/*__*/selection.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append(dict(pairing=d["pairing"], replicate=d["replicate"], variant=d["variant"],
                         weighting=d["weighting"], chosen_setting=d["chosen_setting"],
                         n_parameters=[c for c in d["candidates"] if c["setting"] == d["chosen_setting"]][0][
                             "n_parameters"],
                         selected_fit_seconds=d["selected_fit_seconds"], search_fit_seconds=d["search_fit_seconds"],
                         scoring_seconds_per_1000=d["scoring_seconds_per_1000"],
                         epochs=[c for c in d["candidates"] if c["setting"] == d["chosen_setting"]][0]["epochs"],
                         peak_working_set_gib=d["peak_working_set_gib"]))
    df = pd.DataFrame(rows)
    # The fitting process caches the TF-IDF vocabulary and the transformed messages of a replicate. Only the first
    # weighting run of the text model ("none", always fitted first in its process) includes building them, so the
    # text model's search and scoring times are taken from that run for both weightings.
    text = df.variant.eq("tfidf_logistic")
    first = df[text & df.weighting.eq("none")].set_index(["pairing", "replicate"])
    df["timing_source"] = "own run"
    for col in ("search_fit_seconds", "scoring_seconds_per_1000"):
        df[f"recorded_{col}"] = df[col]
        df.loc[text, col] = [first.loc[(p, r), col] for p, r in zip(df.pairing[text], df.replicate[text])]
    df.loc[text, "timing_source"] = "first weighting run (includes vocabulary and message transformation)"
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    root = a.project.resolve()
    runs = root / "evidence" / "runs"
    out = root / "evidence" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    metrics = collect(runs)
    metrics.to_parquet(out / "run_metrics.parquet", index=False)
    flat = metrics.copy()
    for c in ("recall_ci", "fpr_ci"):
        if c in flat:
            flat[c] = flat[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                                  else json.dumps([float(x) for x in v]))
    flat.to_csv(out / "run_metrics.csv", index=False)
    conf = confirmatory(runs, metrics, a.n_boot)
    conf.to_csv(out / "confirmatory_tests.csv", index=False)
    low = low_prevalence(runs, metrics)
    low.to_csv(out / "low_prevalence_subsets.csv", index=False)
    imp, att = importance_tables(runs)
    imp.to_csv(out / "permutation_importance.csv", index=False)
    att.to_csv(out / "cls_attention.csv", index=False)
    if len(att):
        ftt = imp[imp.variant.eq("feature_token_transformer") & imp.population.eq("in_distribution")]
        corr = []
        for (rep, w), g in ftt.groupby(["replicate", "weighting"]):
            at = att[att.replicate.eq(rep) & att.weighting.eq(w)].set_index("feature")["attention"]
            gi = g.set_index("feature")["mean_drop"]
            rho = spearmanr(at.loc[gi.index], gi).statistic
            corr.append(dict(replicate=rep, weighting=w, spearman_attention_vs_importance=float(rho)))
        pd.DataFrame(corr).to_csv(out / "attention_importance_correlation.csv", index=False)
    compute_table(runs).to_csv(out / "compute_cost.csv", index=False)
    manifest = pd.read_csv(root / "datasets" / "corpus_manifest.csv.gz")
    units = manifest[manifest.status.eq("included")][["record_id", "html_only", "url_count", "words", "attachments",
                                                       "year_header"]]
    error_profile(runs, metrics, units).to_csv(out / "error_profile.csv", index=False)
    print(conf[["population", "model_a", "model_b", "delta", "ci_low", "ci_high", "p_value", "holm_p"]].round(4))


if __name__ == "__main__":
    main()
