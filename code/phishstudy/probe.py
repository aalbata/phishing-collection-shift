"""Collection-identity probes: can the same inputs tell two collections of the SAME class apart?

For each class, a classifier is trained to separate its two collections (the
target is the collection, not phishing). High discrimination means the inputs
carry collection signatures that a phishing detector trained on a
source-labelled pairing can exploit.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .experiment import _frame, wait_for_memory, write_json
from .models import Classifier, TextFeatures
from .stats import ranking

PROBES = {"phishing_collections": ("nazario_recent", "phishing_pot"),
          "legitimate_collections": ("spamassassin_ham", "enron"),
          # protocol amendment 1: era within the mailing-list genre, and corporate against list mail
          "legitimate_era": ("spamassassin_ham", "apache_lists"),
          "legitimate_corporate_vs_lists": ("enron", "apache_lists")}
PROBE_VARIANTS = ("gradient_boosting", "gb_format_only", "tfidf_logistic")


def run_probe(units: pd.DataFrame, splits: pd.DataFrame, seeds: dict, out_root: Path, threads=3, log=print):
    rows = []
    for probe, (a, b) in PROBES.items():
        for rep, seed in seeds.items():
            sp = splits[splits.replicate.eq(rep)].set_index("record_id")["split"]
            part = units[units.collection.isin([a, b])].copy()
            part["label"] = part.collection.eq(b).astype(int)  # target: second collection
            part["split"] = part.record_id.map(sp)
            fr = {k: _frame(part[part.split.eq(k)], f"{probe}_{k}") for k in ("fit", "tune", "test")}
            text = None
            with threadpool_limits(limits=threads):
                for variant in PROBE_VARIANTS:
                    dest = Path(out_root) / probe / f"rep{rep}" / variant
                    if (dest / "metrics.json").exists():
                        continue
                    wait_for_memory(0.8, log)
                    if variant == "tfidf_logistic" and text is None:
                        text = TextFeatures(fr["fit"])
                    best = None
                    for setting in (0, 1):
                        m = Classifier(variant, setting, "balanced", int(seed), threads=threads, text=text)
                        m.fit(fr["fit"], fr["tune"])
                        if best is None or m.tune_ap > best.tune_ap:
                            best = m
                    s = best.score(fr["test"])
                    dest.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(dict(record_id=fr["test"].record_id, collection=fr["test"].collection,
                                      target=fr["test"].label, score=s)).to_parquet(dest / "scores.parquet",
                                                                                    index=False)
                    r = ranking(fr["test"].label.to_numpy(), s)
                    r.update(probe=probe, replicate=rep, variant=variant, positive_collection=b,
                             negative_collection=a, chosen_setting=best.setting, tune_ap=best.tune_ap)
                    write_json(dest / "metrics.json", r)
                    log(f"probe {probe} rep{rep} {variant}: AP={r['average_precision']:.4f} "
                        f"AUC={r['roc_auc']:.4f} (prevalence {r['prevalence']:.3f})")
                    del best
                    gc.collect()
            del text
    for f in sorted(Path(out_root).glob("*/rep*/*/metrics.json")):
        import json
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)
