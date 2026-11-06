"""Write evidence/reports/Experiment_Report.html, a self-contained page built only from evidence files."""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import pandas as pd  # noqa: E402

LABEL = {"logistic": "Logistic regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
         "mlp": "MLP", "single_token_attention": "Single-token attention", "feature_token_transformer": "Feature-token Transformer",
         "ftt_no_url": "FTT without link features", "ftt_no_urgency": "FTT without urgency count",
         "gb_format_only": "Gradient boosting, format only", "tfidf_logistic": "TF-IDF logistic"}
POPS = ["in_distribution", "phishing_shift", "legitimate_shift", "both_shift", "modern_legitimate",
        "phishing_shift_modern_legitimate"]

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--rule:#d9d8d4;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--rule:#383835;--accent:#3987e5}}
body{background:var(--bg);color:var(--ink);font:15px/1.55 Georgia,'Times New Roman',serif;max-width:1080px;margin:0 auto;padding:24px 16px}
h1{font-size:1.6em;margin:.2em 0 .4em}h2{font-size:1.2em;margin-top:1.8em;border-bottom:1px solid var(--rule);padding-bottom:.2em}
p.note{color:var(--muted)}table{border-collapse:collapse;font:12.5px/1.35 system-ui,Segoe UI,Arial,sans-serif;margin:.6em 0}
th,td{padding:4px 8px;border-bottom:1px solid var(--rule);text-align:right}th:first-child,td:first-child{text-align:left}
thead th{border-bottom:2px solid var(--ink)}.wrap{overflow-x:auto}img{max-width:100%;height:auto;background:#fff;border:1px solid var(--rule)}
code{font-size:.92em}
"""


def table(df: pd.DataFrame, digits=3) -> str:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return '<div class="wrap">' + out.to_html(index=False, border=0, escape=True) + "</div>"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    root = p.parse_args().project.resolve()
    M, F = root / "evidence" / "metrics", root / "evidence" / "figures"
    summary = json.loads((root / "datasets" / "corpus_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "audit" / "Numeric_Evidence_Checks.json").read_text(encoding="utf-8"))
    env = json.loads((root / "evidence" / "environment.json").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(M / "run_metrics.parquet")
    sel = metrics[metrics.selected]
    parts = [f"<title>Experiment report</title><style>{CSS}</style>",
             "<h1>Phishing detection across public mail collections: experiment report</h1>",
             "<p class='note'>Every number on this page is read from the saved evidence files of the executed study "
             "(scores of every evaluation unit, recomputed metrics and the independent audit). Message text is never "
             "shown. Results describe these public collections, not a deployed mail service.</p>"]
    parts.append("<h2>Environment</h2>" + table(pd.DataFrame([
        {"item": "platform", "value": env["platform"]}, {"item": "processor", "value": env["processor"]},
        {"item": "logical processors", "value": env["logical_processors"]},
        {"item": "memory (GiB)", "value": f"{env['memory_total_gib']:.2f}"},
        {"item": "python", "value": env["python"].split()[0]}] +
        [{"item": k, "value": v} for k, v in env["packages"].items()])))
    units = pd.Series(summary["units_per_collection"]).rename("units").reset_index().rename(columns={"index": "collection"})
    parts.append("<h2>Analysis units</h2>" + table(units) +
                 f"<p>Eligible messages {summary['eligible']:,}; duplicate groups {summary['groups']:,}; "
                 f"label-conflict groups {summary['label_conflict_groups']}; cross-collection groups "
                 f"{summary['cross_collection_groups']}.</p>")
    defs = {"in_distribution": "test units of the training phishing and legitimate collections",
            "phishing_shift": "all units of the other phishing collection; test units of the training legitimate collection",
            "legitimate_shift": "test units of the training phishing collection; all units of the other legitimate collection",
            "both_shift": "all units of the other phishing and the other legitimate collection",
            "modern_legitimate": "test units of the training phishing collection; all Apache list units",
            "phishing_shift_modern_legitimate": "all units of the other phishing collection; all Apache list units"}
    sizes = (sel[sel.alpha.isna()].groupby(["pairing", "population"])[["n_positive", "n_negative"]]
             .agg(["min", "max"]).reset_index())
    sizes.columns = ["pairing", "population", "phishing min", "phishing max", "legitimate min", "legitimate max"]
    sizes["definition"] = sizes.population.map(defs)
    parts.append("<h2>Test populations</h2><p class='note'>Unit counts over the five replicates (test parts vary by "
                 "replicate; collections used whole are constant).</p>" + table(sizes))
    ap = (sel[sel.alpha.isna()].groupby(["pairing", "variant", "population"]).average_precision.mean()
          .unstack("population")[POPS].reset_index())
    ap["variant"] = ap.variant.map(LABEL)
    parts.append("<h2>Average precision by population (mean of five replicates, selected weighting)</h2>" + table(ap))
    op = sel[sel.alpha.eq(0.01)].groupby(["pairing", "variant", "population"])[["recall", "fpr"]].mean().reset_index()
    op = op[op.population.isin(["in_distribution", "legitimate_shift", "modern_legitimate", "both_shift"])]
    op["variant"] = op.variant.map(LABEL)
    parts.append("<h2>Recall and realised false-positive rate at the 1 percent threshold</h2>"
                 "<p class='note'>Thresholds were calibrated on held-out legitimate units of the training collection; "
                 "under shift the rate is measured, not guaranteed.</p>" + table(op, 4))
    conf = pd.read_csv(M / "confirmatory_tests.csv")
    conf["model_a"] = conf.model_a.map(LABEL)
    conf["model_b"] = conf.model_b.map(LABEL)
    parts.append("<h2>Confirmatory comparisons (primary pairing NZ-SA)</h2>" +
                 table(conf[["population", "model_a", "model_b", "delta", "ci_low", "ci_high", "p_value", "holm_p"]], 4))
    if (M / "probe_metrics.csv").exists():
        pr = pd.read_csv(M / "probe_metrics.csv").groupby(["probe", "variant"])[["average_precision", "roc_auc"]] \
            .agg(["mean", "min", "max"]).round(4)
        pr.columns = [" ".join(c) for c in pr.columns]
        parts.append("<h2>Collection-identity probes</h2>" + table(pr.reset_index(), 4))
    for png in sorted(F.glob("Figure_*.png")):
        data = base64.b64encode(png.read_bytes()).decode()
        parts.append(f"<h2>{html.escape(png.stem.replace('_', ' '))}</h2><img alt='{html.escape(png.stem)}' "
                     f"src='data:image/png;base64,{data}'>")
    parts.append("<h2>Independent numerical audit</h2><pre>" + html.escape(json.dumps(audit, indent=2)) + "</pre>")
    out = root / "evidence" / "reports" / "Experiment_Report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                   "<meta name='viewport' content='width=device-width,initial-scale=1'>" + parts[0] +
                   "</head><body>" + "".join(parts[1:]) + "</body></html>", encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
