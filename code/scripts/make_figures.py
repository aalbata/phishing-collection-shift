"""Draw every figure from evidence files (metrics, saved scores, corpus summary) into evidence/figures/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from phishstudy.analysis import population  # noqa: E402
from phishstudy.figures import FEATURE_LABEL, GRID, INK, MM, MODEL_COLOR, MODEL_LABEL, MUTED, save, style, workflow  # noqa: E402

PAIRING_ORDER = ["NZ-SA", "NZ-EN", "PP-SA", "PP-EN"]
POP_ORDER = ["in_distribution", "phishing_shift", "legitimate_shift", "both_shift", "modern_legitimate",
             "phishing_shift_modern_legitimate"]
POP_SHORT = {"in_distribution": "ID", "phishing_shift": "P′", "legitimate_shift": "L′", "both_shift": "P′+L′",
             "modern_legitimate": "AL", "phishing_shift_modern_legitimate": "P′+AL"}
VARIANT_ORDER = ["logistic", "random_forest", "gradient_boosting", "gb_format_only", "mlp", "single_token_attention",
                 "feature_token_transformer", "ftt_no_url", "ftt_no_urgency", "tfidf_logistic"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def fig_transfer(metrics: pd.DataFrame, out: Path, value="average_precision", stem="Figure_2_Transfer_AP",
                 label="Average precision (mean of 5 replicates)", vmin=0.0, vmax=1.0):
    sel = metrics[metrics.selected & metrics.alpha.isna()]
    mean = sel.groupby(["variant", "pairing", "population"])[value].mean()
    cols = [(p, q) for p in PAIRING_ORDER for q in POP_ORDER]
    M = np.array([[mean.get((v, p, q), np.nan) for (p, q) in cols] for v in VARIANT_ORDER])
    style()
    fig, ax = plt.subplots(figsize=(155 * MM, 82 * MM))
    im = ax.imshow(M, cmap=BLUES, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                        color="white" if M[i, j] > vmin + 0.62 * (vmax - vmin) else INK)
    ax.set_xticks(range(len(cols)), [POP_SHORT[q] for _, q in cols], fontsize=6.5, rotation=90)
    ax.set_yticks(range(len(VARIANT_ORDER)), [MODEL_LABEL[v] for v in VARIANT_ORDER], fontsize=6.5)
    for k in range(1, len(PAIRING_ORDER)):
        ax.axvline(k * len(POP_ORDER) - 0.5, color="white", linewidth=2)
    for k, p in enumerate(PAIRING_ORDER):
        ax.text(k * len(POP_ORDER) + (len(POP_ORDER) - 1) / 2, -0.9, f"trained on {p}", ha="center", va="bottom",
                fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label, fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)
    return save(fig, out, stem)


def fig_fpr_transfer(metrics: pd.DataFrame, out: Path, alpha=0.01, stem="Figure_3_FPR_Transfer"):
    """Realised FPR at the alpha threshold on the training collection's test units and on unseen legitimate mail."""
    sel = metrics[metrics.selected & metrics.alpha.eq(alpha) & metrics.population.isin(
        ["in_distribution", "legitimate_shift", "modern_legitimate"])]
    shows = ["logistic", "random_forest", "gradient_boosting", "mlp", "single_token_attention",
             "feature_token_transformer", "tfidf_logistic"]
    style()
    fig, axes = plt.subplots(1, 4, figsize=(160 * MM, 64 * MM), sharey=True)
    markers = {"in_distribution": "o", "legitimate_shift": "s", "modern_legitimate": "^"}
    names = {"in_distribution": "test units of L", "legitimate_shift": "all units of L′",
             "modern_legitimate": "all AL units"}
    for ax, pairing in zip(axes, PAIRING_ORDER):
        z = sel[sel.pairing.eq(pairing)]
        for i, v in enumerate(shows):
            for k, pop in enumerate(markers):
                r = z[z.variant.eq(v) & z.population.eq(pop)].fpr.dropna()
                if not len(r):
                    continue
                y = i + (k - 1) * 0.22
                vals = np.clip(r.to_numpy(), 1e-5, 1)
                ax.plot([vals.min(), vals.max()], [y, y], color=MODEL_COLOR[v], linewidth=0.8)
                ax.plot(vals.mean(), y, markers[pop], color=MODEL_COLOR[v], markersize=3.2,
                        markerfacecolor="white" if pop != "in_distribution" else MODEL_COLOR[v], markeredgewidth=0.7)
        ax.axvline(alpha, color=MUTED, linestyle="--", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1.2)
        ax.set_xticks([1e-4, 1e-2, 1], ["0.01%", "1%", "100%"])
        ax.set_xticks([1e-5, 1e-3, 1e-1], minor=True)
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(f"trained on {pairing}", fontsize=7)
        ax.set_xlabel("False-positive rate (log scale)")
        ax.grid(axis="x", color=GRID, linewidth=0.4)
    axes[0].set_yticks(range(len(shows)), [MODEL_LABEL[v] for v in shows])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker=markers[p], linestyle="none", color=MUTED,
                          markerfacecolor="white" if p != "in_distribution" else MUTED, markersize=3.5,
                          label=names[p]) for p in markers]
    handles.append(plt.Line2D([], [], linestyle="--", color=MUTED, linewidth=0.7, label=f"{100 * alpha:g}% budget"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.0), fontsize=6.5)
    fig.subplots_adjust(wspace=0.16, bottom=0.27)
    return save(fig, out, stem)


def fig_recall_shift(metrics: pd.DataFrame, out: Path, alpha=0.01, stem="Figure_4_Recall_Shift"):
    sel = metrics[metrics.selected & metrics.alpha.eq(alpha)]
    shows = ["gradient_boosting", "gb_format_only", "mlp", "single_token_attention", "feature_token_transformer",
             "tfidf_logistic"]
    style()
    fig, axes = plt.subplots(1, 4, figsize=(160 * MM, 60 * MM), sharey=True)
    pops = ["in_distribution", "phishing_shift", "legitimate_shift", "both_shift"]
    for ax, pairing in zip(axes, PAIRING_ORDER):
        z = sel[sel.pairing.eq(pairing)]
        for v in shows:
            m = [z[z.variant.eq(v) & z.population.eq(p)].recall.mean() for p in pops]
            lo = [z[z.variant.eq(v) & z.population.eq(p)].recall.min() for p in pops]
            hi = [z[z.variant.eq(v) & z.population.eq(p)].recall.max() for p in pops]
            x = np.arange(len(pops))
            ax.fill_between(x, lo, hi, color=MODEL_COLOR[v], alpha=0.12, linewidth=0)
            ax.plot(x, m, marker="o", markersize=2.4, color=MODEL_COLOR[v], label=MODEL_LABEL[v])
        ax.set_xticks(range(len(pops)), [POP_SHORT[p] for p in pops])
        ax.set_ylim(0, 1.02)
        ax.set_title(f"trained on {pairing}", fontsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.4)
    axes[0].set_ylabel(f"Recall at the {100 * alpha:g}% budget threshold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    fig.subplots_adjust(wspace=0.16, bottom=0.24)
    return save(fig, out, stem)


def fig_probes(probes: pd.DataFrame, out: Path, stem="Figure_5_Collection_Probes"):
    order = ["phishing_collections", "legitimate_collections", "legitimate_era", "legitimate_corporate_vs_lists"]
    names = {"phishing_collections": "NZ vs PP\n(phishing)", "legitimate_collections": "SA vs EN\n(legitimate)",
             "legitimate_era": "SA vs AL\n(2002 vs 2015-2025)",
             "legitimate_corporate_vs_lists": "EN vs AL\n(corporate vs lists)"}
    variants = ["gradient_boosting", "gb_format_only", "tfidf_logistic"]
    markers = {"gradient_boosting": "o", "gb_format_only": "s", "tfidf_logistic": "^"}
    style()
    fig, ax = plt.subplots(figsize=(120 * MM, 55 * MM))
    w = 0.22
    lows = []
    for k, v in enumerate(variants):
        m = np.array([probes[probes.probe.eq(p) & probes.variant.eq(v)].roc_auc.mean() for p in order])
        lo = np.array([probes[probes.probe.eq(p) & probes.variant.eq(v)].roc_auc.min() for p in order])
        hi = np.array([probes[probes.probe.eq(p) & probes.variant.eq(v)].roc_auc.max() for p in order])
        lows.append(lo.min())
        x = np.arange(len(order)) + (k - 1) * w
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt=markers[v], color=MODEL_COLOR[v], markersize=4,
                    elinewidth=0.8, capsize=2, label=MODEL_LABEL[v])
    bottom = np.floor(min(lows) * 20) / 20 - 0.05
    ax.set_ylim(bottom, 1.005)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks(range(len(order)), [names[p] for p in order], fontsize=6.5)
    ax.set_ylabel("ROC-AUC for telling the two\ncollections apart (test units)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    return save(fig, out, stem)


def fig_pr(runs: Path, metrics: pd.DataFrame, out: Path, pairing="NZ-SA", rep=0, stem="Figure_S1_PR_Curves"):
    from sklearn.metrics import precision_recall_curve
    sel = metrics[metrics.selected & metrics.alpha.isna() & metrics.pairing.eq(pairing) & metrics.replicate.eq(rep)]
    chosen = sel.drop_duplicates("variant").set_index("variant")["weighting"]
    shows = ["gradient_boosting", "mlp", "single_token_attention", "feature_token_transformer", "tfidf_logistic"]
    pops = ["in_distribution", "both_shift", "phishing_shift_modern_legitimate"]
    titles = {"in_distribution": "in-distribution (NZ + SA test)", "both_shift": "both shift (PP + EN)",
              "phishing_shift_modern_legitimate": "PP + Apache lists"}
    style()
    fig, axes = plt.subplots(1, 3, figsize=(160 * MM, 60 * MM), sharey=True)
    styles = ["-", "--", "-.", "-", ":"]
    for ax, pop in zip(axes, pops):
        for v, ls in zip(shows, styles):
            sc = pd.read_parquet(runs / pairing / f"rep{rep}" / f"{v}__{chosen[v]}" / "scores.parquet")
            z = population(sc, pop)
            p, r, _ = precision_recall_curve(z.label, z.score)
            ax.step(r, p, where="post", linestyle=ls, color=MODEL_COLOR[v], linewidth=0.9, label=MODEL_LABEL[v])
        prev = float(z.label.mean())
        ax.axhline(prev, color=MUTED, linestyle=":", linewidth=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_title(titles[pop], fontsize=7)
        ax.grid(color=GRID, linewidth=0.4)
    axes[0].set_ylabel("Precision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    fig.subplots_adjust(wspace=0.2, bottom=0.27)
    return save(fig, out, stem)


def fig_importance(imp: pd.DataFrame, att: pd.DataFrame, out: Path, stem="Figure_6_Importance_Attention"):
    ftt = imp[imp.variant.eq("feature_token_transformer")]
    gb = imp[imp.variant.eq("gradient_boosting")]
    order = (ftt[ftt.population.eq("in_distribution")].groupby("feature").mean_drop.mean()
             .sort_values(ascending=False).index[:12].tolist())
    style()
    fig, axes = plt.subplots(1, 3, figsize=(160 * MM, 66 * MM), sharey=True)
    y = np.arange(len(order))
    for ax, (name, data) in zip(axes[:2], [("Feature-token Transformer", ftt), ("Gradient boosting", gb)]):
        for k, (pop, col) in enumerate([("in_distribution", MODEL_COLOR["feature_token_transformer"]),
                                         ("both_shift", MODEL_COLOR["mlp"])]):
            d = data[data.population.eq(pop)].groupby("feature").mean_drop
            m, lo, hi = d.mean().reindex(order), d.min().reindex(order), d.max().reindex(order)
            ax.barh(y + (k - 0.5) * 0.38, m, height=0.36, color=col,
                    label="in-distribution" if pop == "in_distribution" else "both shift")
            ax.errorbar(m, y + (k - 0.5) * 0.38, xerr=[m - lo, hi - m], fmt="none", ecolor=INK, elinewidth=0.5,
                        capsize=1)
        ax.set_title(name, fontsize=7)
        ax.set_xlabel("Decrease in average precision\nwhen the feature is shuffled")
        ax.grid(axis="x", color=GRID, linewidth=0.4)
        ax.axvline(0, color=MUTED, linewidth=0.5)
    a = att.groupby("feature").attention
    m, lo, hi = a.mean().reindex(order), a.min().reindex(order), a.max().reindex(order)
    axes[2].barh(y, m, height=0.5, color=MODEL_COLOR["feature_token_transformer"])
    axes[2].errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor=INK, elinewidth=0.5, capsize=1)
    axes[2].axvline(1 / (att.feature.nunique() + 1), color=MUTED, linestyle=":", linewidth=0.7)
    axes[2].set_title("[CLS] attention, last block", fontsize=7)
    axes[2].set_xlabel("Mean attention weight\n(dotted line, uniform weight)")
    axes[2].grid(axis="x", color=GRID, linewidth=0.4)
    axes[0].set_yticks(y, [FEATURE_LABEL[f] for f in order])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=6.5)
    fig.subplots_adjust(wspace=0.1)
    return save(fig, out, stem)


def selected_only(imp: pd.DataFrame, att: pd.DataFrame, metrics: pd.DataFrame):
    """Keep the fit of each replicate and variant whose weighting was selected on the tune part."""
    first = (metrics[metrics.alpha.isna() & metrics.population.eq("in_distribution") & metrics.pairing.eq("NZ-SA")]
             .drop_duplicates(["replicate", "variant"])[["replicate", "variant", "selected_weighting"]])
    imp = imp.merge(first, on=["replicate", "variant"])
    imp = imp[imp.weighting.eq(imp.selected_weighting)]
    ftt = first[first.variant.eq("feature_token_transformer")][["replicate", "selected_weighting"]]
    att = att.merge(ftt, on="replicate")
    att = att[att.weighting.eq(att.selected_weighting)]
    return imp, att


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    a = p.parse_args()
    root = a.project.resolve()
    out = root / "evidence" / "figures"
    M = root / "evidence" / "metrics"
    summary = json.loads((root / "datasets" / "corpus_summary.json").read_text(encoding="utf-8"))
    made = workflow(summary, out)
    metrics = pd.read_parquet(M / "run_metrics.parquet")
    made += fig_transfer(metrics, out)
    made += fig_transfer(metrics, out, value="roc_auc", stem="Figure_S2_Transfer_ROC_AUC",
                         label="ROC-AUC (mean of 5 replicates)", vmin=0.5)
    made += fig_fpr_transfer(metrics, out)
    made += fig_recall_shift(metrics, out)
    if (M / "probe_metrics.csv").exists():
        made += fig_probes(pd.read_csv(M / "probe_metrics.csv"), out)
    made += fig_pr(root / "evidence" / "runs", metrics, out)
    imp, att = pd.read_csv(M / "permutation_importance.csv"), pd.read_csv(M / "cls_attention.csv")
    if len(imp):
        made += fig_importance(*selected_only(imp, att, metrics), out)
    for f in made:
        print(f.relative_to(root))


if __name__ == "__main__":
    main()
