"""Independent numerical audit of every stored result; writes audit/Numeric_Evidence_Checks.json.

Uses only phishstudy.verify for arithmetic (it shares no metric code with the analysis) and stops on the first
mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phishstudy import verify  # noqa: E402

PAIRINGS = {"NZ-SA": ("nazario_recent", "spamassassin_ham"), "NZ-EN": ("nazario_recent", "enron"),
            "PP-SA": ("phishing_pot", "spamassassin_ham"), "PP-EN": ("phishing_pot", "enron")}
OTHER = {"nazario_recent": "phishing_pot", "phishing_pot": "nazario_recent",
         "spamassassin_ham": "enron", "enron": "spamassassin_ham"}


def bootstrap_independent(reps, n_boot, seed):
    """Same resampling scheme as the analysis, re-implemented with verify.ap on expanded samples."""
    rng = np.random.default_rng(seed)
    observed = float(np.mean([verify.ap(y, a) - verify.ap(y, b) for y, a, b in reps]))
    idx = [(np.flatnonzero(y == 1), np.flatnonzero(y == 0)) for y, _, _ in reps]
    draws = np.empty(n_boot)
    for i in range(n_boot):
        vals = []
        for r in rng.integers(0, len(reps), size=len(reps)):
            y, a, b = reps[r]
            pos, neg = idx[r]
            take = np.concatenate([rng.choice(pos, size=len(pos), replace=True),
                                   rng.choice(neg, size=len(neg), replace=True)])
            vals.append(verify.ap(y[take], a[take]) - verify.ap(y[take], b[take]))
        draws[i] = np.mean(vals)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return observed, float(lo), float(hi), float(min(1.0, max(p, 1 / n_boot)))


def holm_independent(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def _ranks(x: np.ndarray) -> np.ndarray:
    """Mid-ranks (1-based) by direct comparison."""
    x = np.asarray(x, dtype=float)
    less = (x[None, :] < x[:, None]).sum(axis=1)
    equal = (x[None, :] == x[:, None]).sum(axis=1)
    return less + (equal + 1) / 2


def _spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def descriptive_tables(root, runs, metrics_dir, metrics, populations, weighting_of) -> dict:
    """Recompute the descriptive result files from saved scores, unit features and run records."""
    out = {}

    def scores(pairing, rep, variant, weighting):
        return pd.read_parquet(runs / pairing / f"rep{rep}" / f"{variant}__{weighting}" / "scores.parquet")

    def threshold(pairing, rep, variant, weighting, population, alpha):
        m = metrics[(metrics.pairing == pairing) & (metrics.replicate == rep) & (metrics.variant == variant)
                    & (metrics.weighting == weighting) & (metrics.population == population) & (metrics.alpha == alpha)]
        verify.check(len(m) == 1, f"no unique operating point for {pairing} rep{rep} {variant} {population}")
        return float(m.threshold.iloc[0])

    # subsets with 1 percent phishing: the same seed and draw order, own AP and precision arithmetic
    low = pd.read_csv(metrics_dir / "low_prevalence_subsets.csv")
    rng = np.random.default_rng(20261105)
    for _, r in low.iterrows():
        w = weighting_of(r.pairing, r.replicate, r.variant)
        sc = scores(r.pairing, r.replicate, r.variant, w)
        pf, nf = populations[r.population]
        pos = sc[(sc.frame == pf) & (sc.label == 1)]
        neg = sc[(sc.frame == nf) & (sc.label == 0)]
        k = int(round(0.01 / 0.99 * len(neg)))
        verify.check(k == r.n_positive_subset and len(neg) == r.n_negative, f"subset size {r.pairing} {r.variant}")
        t = threshold(r.pairing, r.replicate, r.variant, w, r.population, 0.01)
        aps, precs = [], []
        ys_neg, ss_neg = np.zeros(len(neg)), neg.score.to_numpy()
        for _ in range(int(r.draws)):
            take = rng.choice(len(pos), size=k, replace=False)
            y = np.concatenate([np.ones(k), ys_neg])
            s = np.concatenate([pos.score.to_numpy()[take], ss_neg])
            aps.append(verify.ap(y, s))
            flagged = s > t
            precs.append(np.nan if flagged.sum() == 0 else float(y[flagged].sum() / flagged.sum()))
        for name, mine in (("ap_mean", np.mean(aps)), ("ap_q025", np.quantile(aps, .025)),
                           ("ap_q975", np.quantile(aps, .975)), ("precision_mean", np.nanmean(precs)),
                           ("precision_q025", np.nanquantile(precs, .025)),
                           ("precision_q975", np.nanquantile(precs, .975))):
            verify.check(verify.close(mine, r[name]), f"low-prevalence {name} mismatch {r.pairing} {r.variant}")
    out["low_prevalence_rows"] = int(len(low))
    # permutation importance and attention tables, and their rank correlation
    imp = pd.read_csv(metrics_dir / "permutation_importance.csv")
    att = pd.read_csv(metrics_dir / "cls_attention.csv")
    imp_idx = imp.set_index(["replicate", "variant", "weighting", "population", "feature"])
    att_idx = att.set_index(["replicate", "weighting", "feature"])
    n_imp = n_att = 0
    for f in sorted(runs.glob("NZ-SA/rep*/*__*/permutation_importance.json")):
        variant, weighting = f.parent.name.split("__")
        rep = int(f.parent.parent.name[3:])
        d = json.loads(f.read_text(encoding="utf-8"))
        for pop in ("in_distribution", "both_shift"):
            for feat, v in d[pop]["importance"].items():
                row = imp_idx.loc[(rep, variant, weighting, pop, feat)]
                verify.check(verify.close(row.mean_drop, v["mean"]) and verify.close(row.sd_drop, v["sd"]),
                             f"importance table differs for {f} {pop} {feat}")
                n_imp += 1
        a = f.parent / "cls_attention.json"
        if a.exists():
            for feat, v in json.loads(a.read_text(encoding="utf-8"))["weights"].items():
                verify.check(verify.close(att_idx.loc[(rep, weighting, feat)].attention, v), f"attention differs {a}")
                n_att += 1
    verify.check(n_imp == len(imp) and n_att == len(att), "importance or attention table has extra rows")
    corr = pd.read_csv(metrics_dir / "attention_importance_correlation.csv")
    for _, c in corr.iterrows():
        g = imp[(imp.variant == "feature_token_transformer") & (imp.population == "in_distribution")
                & (imp.replicate == c.replicate) & (imp.weighting == c.weighting)].set_index("feature").mean_drop
        at = att[(att.replicate == c.replicate) & (att.weighting == c.weighting)].set_index("feature").attention
        verify.check(verify.close(_spearman(at.loc[g.index].to_numpy(), g.to_numpy()),
                                  c.spearman_attention_vs_importance), "attention-importance correlation mismatch")
    out.update(importance_rows=n_imp, attention_rows=n_att, correlations=int(len(corr)))
    # error profile at the 1 percent threshold of the primary pairing
    man = pd.read_csv(root / "datasets" / "corpus_manifest.csv.gz",
                      usecols=["record_id", "status", "html_only", "url_count", "words"])
    feats = man[man.status.eq("included")].set_index("record_id")
    err = pd.read_csv(metrics_dir / "error_profile.csv")
    n_err = 0
    for (rep, variant, pop), g in err.groupby(["replicate", "variant", "population"]):
        w = weighting_of("NZ-SA", rep, variant)
        t = threshold("NZ-SA", rep, variant, w, pop, 0.01)
        sc = scores("NZ-SA", rep, variant, w)
        pf, nf = populations[pop]
        z = pd.concat([sc[(sc.frame == pf) & (sc.label == 1)], sc[(sc.frame == nf) & (sc.label == 0)]])
        f_ = feats.loc[z.record_id]
        words, urls, html = f_.words.to_numpy(), f_.url_count.to_numpy(), f_.html_only.to_numpy()
        level = {"length": np.where(words <= 49, "<50 words", np.where(words <= 199, "50-199 words", ">=200 words")),
                 "links": np.where(urls > 0, "has links", "no links"),
                 "format": np.where(html == 1, "HTML only", "plain text present")}
        flagged, label = z.score.to_numpy() > t, z.label.to_numpy()
        for _, r in g.iterrows():
            mask = (level[r.property] == r.level) & (label == r.label)
            verify.check(int(mask.sum()) == r.n and int(flagged[mask].sum()) == r.flagged,
                         f"error profile counts differ {rep} {variant} {pop} {r.property} {r.level}")
            n_err += 1
        verify.check(len(g) == sum(len(set(zip(level[p], label))) for p in level), "error profile rows missing")
    out["error_profile_rows"] = n_err
    # compute cost table equals the run records
    cost = pd.read_csv(metrics_dir / "compute_cost.csv")
    for _, r in cost.iterrows():
        def record(weighting):
            return json.loads((runs / r.pairing / f"rep{r.replicate}" / f"{r.variant}__{weighting}" / "selection.json")
                              .read_text(encoding="utf-8"))
        own = record(r.weighting)
        # the text model's times come from its first ("none") run, which built the vocabulary and transformed text
        timing = record("none") if r.variant == "tfidf_logistic" else own
        verify.check(verify.close(r.selected_fit_seconds, own["selected_fit_seconds"])
                     and verify.close(r.search_fit_seconds, timing["search_fit_seconds"])
                     and verify.close(r.scoring_seconds_per_1000, timing["scoring_seconds_per_1000"]),
                     f"compute cost differs for {r.pairing} {r.replicate} {r.variant}")
    out["compute_rows"] = int(len(cost))
    return out


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--project", type=Path, default=CODE.parent)
    ap_.add_argument("--skip-bootstrap", action="store_true")
    a = ap_.parse_args()
    root = a.project.resolve()
    runs, metrics_dir = root / "evidence" / "runs", root / "evidence" / "metrics"
    manifest = pd.read_csv(root / "datasets" / "corpus_manifest.csv.gz", usecols=["record_id", "collection", "label", "status"])
    units = manifest[manifest.status.eq("included")].drop(columns="status").reset_index(drop=True)
    splits = pd.read_csv(root / "datasets" / "splits.csv.gz")
    metrics = pd.read_parquet(metrics_dir / "run_metrics.parquet")
    report = dict(utc=datetime.now(timezone.utc).isoformat(), checks={})
    # 1. scores, memberships, thresholds, every metric row
    report["checks"]["runs"] = verify.audit_runs(runs, units, splits, PAIRINGS, OTHER, metrics)
    # 2. split disjointness per replicate and collection
    for rep, part in splits.groupby("replicate"):
        verify.check(part.record_id.is_unique, f"replicate {rep} assigns a unit twice")
        verify.check(set(part.record_id) == set(units.record_id), f"replicate {rep} does not cover the units")
    report["checks"]["splits"] = dict(replicates=int(splits.replicate.nunique()), units=int(len(units)))
    # 3. confirmatory tests
    conf = pd.read_csv(metrics_dir / "confirmatory_tests.csv")
    sel = metrics[metrics.alpha.isna() & metrics.population.eq("in_distribution")]
    chosen = (sel.drop_duplicates(["pairing", "replicate", "variant"])
              .set_index(["pairing", "replicate", "variant"])["selected_weighting"])
    from phishstudy.analysis import POPULATIONS  # population membership definition only
    checked = 0
    for _, row in conf.iterrows():
        reps = []
        for rep in sorted(metrics[metrics.pairing.eq(row.pairing)].replicate.unique()):
            def pop(variant):
                sc = pd.read_parquet(runs / row.pairing / f"rep{rep}" /
                                     f"{variant}__{chosen[(row.pairing, rep, variant)]}" / "scores.parquet")
                pf, nf = POPULATIONS[row.population]
                z = pd.concat([sc[(sc.frame == pf) & (sc.label == 1)], sc[(sc.frame == nf) & (sc.label == 0)]])
                return z.sort_values("record_id")
            za, zb = pop(row.model_a), pop(row.model_b)
            verify.check((za.record_id.to_numpy() == zb.record_id.to_numpy()).all(), "unpaired test populations")
            reps.append((za.label.to_numpy(), za.score.to_numpy(), zb.score.to_numpy()))
        if not a.skip_bootstrap:
            d, lo, hi, p = bootstrap_independent(reps, int(row.n_boot), int(row.seed))
            for name, mine, stored in (("delta", d, row.delta), ("ci_low", lo, row.ci_low), ("ci_high", hi, row.ci_high),
                                       ("p_value", p, row.p_value)):
                verify.check(abs(mine - stored) < 1e-9, f"confirmatory {name} mismatch for {row.model_a} vs {row.model_b}")
        else:
            d = float(np.mean([verify.ap(y, s1) - verify.ap(y, s2) for y, s1, s2 in reps]))
            verify.check(abs(d - row.delta) < 1e-9, "confirmatory delta mismatch")
        checked += 1
    adj = holm_independent(conf.p_value.tolist())
    verify.check(all(abs(x - y) < 1e-12 for x, y in zip(adj, conf.holm_p)), "Holm adjustment mismatch")
    report["checks"]["confirmatory_tests"] = dict(tests=checked, bootstrap_recomputed=not a.skip_bootstrap)
    # 4. probes
    probe_rows = 0
    pm = pd.read_csv(metrics_dir / "probe_metrics.csv")
    for _, r in pm.iterrows():
        sc = pd.read_parquet(root / "evidence" / "probes" / r.probe / f"rep{r.replicate}" / r.variant / "scores.parquet")
        verify.check(verify.close(r.average_precision, verify.ap(sc.target, sc.score)), f"probe AP {r.probe}")
        verify.check(verify.close(r.roc_auc, verify.auc(sc.target, sc.score)), f"probe AUC {r.probe}")
        probe_rows += 1
    report["checks"]["probes"] = dict(rows=probe_rows)
    # 5. permutation importance baselines equal the stored population AP of the same fit
    imp_runs = 0
    for f in sorted(runs.glob("NZ-SA/rep*/*__*/permutation_importance.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        variant, weighting = f.parent.name.split("__")
        rep = int(f.parent.parent.name[3:])
        for pop in ("in_distribution", "both_shift"):
            m = metrics[(metrics.pairing == "NZ-SA") & (metrics.replicate == rep) & (metrics.variant == variant)
                        & (metrics.weighting == weighting) & (metrics.population == pop) & metrics.alpha.isna()]
            verify.check(verify.close(m.average_precision.iloc[0], d[pop]["base_average_precision"]),
                         f"permutation baseline differs from stored AP in {f}")
        imp_runs += 1
    report["checks"]["permutation_importance_baselines"] = dict(runs=imp_runs)
    weighting_of = lambda p, r, v: chosen[(p, r, v)]  # noqa: E731
    report["checks"]["descriptive_tables"] = descriptive_tables(root, runs, metrics_dir, metrics, POPULATIONS,
                                                                weighting_of)
    report["status"] = "PASS"
    out = root / "audit" / "Numeric_Evidence_Checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
