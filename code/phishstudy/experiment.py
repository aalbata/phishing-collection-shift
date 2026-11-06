"""Fit every declared model in every training pairing and replicate; save scores first.

A training pairing combines one phishing collection P and one legitimate
collection L. Models are fitted on the fit parts of P and L, selected on their
tune parts and thresholded on the calibration part of L. Scores are saved for
the calibration and test parts of P and L and for every unit of the other
phishing collection P' and the other legitimate collection L'. Metrics are
computed later from these saved scores only.
"""
from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .models import VARIANTS, Classifier, TextFeatures
from .stats import np_threshold

PAIRINGS = {
    "NZ-SA": ("nazario_recent", "spamassassin_ham"),
    "NZ-EN": ("nazario_recent", "enron"),
    "PP-SA": ("phishing_pot", "spamassassin_ham"),
    "PP-EN": ("phishing_pot", "enron"),
}
OTHER = {"nazario_recent": "phishing_pot", "phishing_pot": "nazario_recent",
         "spamassassin_ham": "enron", "enron": "spamassassin_ham"}
MODERN = "apache_lists"  # evaluation-only legitimate collection (protocol amendment 1)
ALPHAS = (0.001, 0.005, 0.01)
DELTA = 0.05
PERMUTATION_VARIANTS = ("logistic", "random_forest", "gradient_boosting", "mlp", "feature_token_transformer")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=_default), encoding="utf-8")
    tmp.replace(path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def available_gib() -> float:
    return psutil.virtual_memory().available / 2**30


def wait_for_memory(min_gib: float, log, max_wait_s: int = 1800) -> None:
    waited = 0
    while available_gib() < min_gib and waited < max_wait_s:
        log(f"available memory {available_gib():.2f} GiB below {min_gib} GiB; pausing")
        time.sleep(30)
        waited += 30


def acquire_lock(dest: Path) -> bool:
    """Claim one fit for this process; a lock left by a process that no longer runs is taken over."""
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / "lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = int(lock.read_text().strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and psutil.pid_exists(owner):
                return False
            lock.unlink(missing_ok=True)
    return False


def _frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.reset_index(drop=True)
    out.attrs["name"] = name
    return out


def pairing_frames(units: pd.DataFrame, splits: pd.DataFrame, pairing: str, replicate: int) -> dict:
    p, l = PAIRINGS[pairing]
    sp = splits[splits.replicate.eq(replicate)].set_index("record_id")["split"]
    frames = {}
    for role, coll in (("P", p), ("L", l)):
        part = units[units.collection.eq(coll)].copy()
        part["split"] = part.record_id.map(sp)
        if part.split.isna().any():
            raise AssertionError(f"unsplit units in {coll}")
        for name in ("fit", "tune", "calibration", "test"):
            frames[f"{role}_{name}"] = part[part.split.eq(name)]
    frames["Pother_all"] = units[units.collection.eq(OTHER[p])].assign(split="all")
    frames["Lother_all"] = units[units.collection.eq(OTHER[l])].assign(split="all")
    frames["Mod_all"] = units[units.collection.eq(MODERN)].assign(split="all")
    out = {
        "fit": _frame(pd.concat([frames["P_fit"], frames["L_fit"]]), "fit"),
        "tune": _frame(pd.concat([frames["P_tune"], frames["L_tune"]]), "tune"),
    }
    for key in ("P_calibration", "L_calibration", "P_test", "L_test", "Pother_all", "Lother_all", "Mod_all"):
        out[key] = _frame(frames[key], key)
    ids = [set(v.record_id) for v in out.values()]
    total = sum(len(s) for s in ids)
    if len(set().union(*ids)) != total:
        raise AssertionError("a unit appears in more than one frame")
    return out


SCORED = ("P_calibration", "L_calibration", "P_test", "L_test", "Pother_all", "Lother_all", "Mod_all")


def permutation_importance(model: Classifier, frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    """Mean decrease in average precision when one feature column is shuffled across messages."""
    # The preprocessing is column-wise, so shuffling a column of the transformed matrix equals
    # transforming a frame whose raw column was shuffled; the matrix is built once.
    y = frame.label.to_numpy()
    X = model._numeric(frame)
    base = average_precision_score(y, model.score_matrix(X))
    rng = np.random.default_rng(seed)
    out = {}
    for j, col in enumerate(model.columns):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(frame)), j]
            drops.append(base - average_precision_score(y, model.score_matrix(Xp)))
        out[col] = dict(mean=float(np.mean(drops)), sd=float(np.std(drops, ddof=1)), repeats=repeats)
    return dict(base_average_precision=float(base), n=int(len(frame)), importance=out)


def cls_attention(model: Classifier, frame: pd.DataFrame) -> dict | None:
    """Mean attention weight from the [CLS] query to each feature token in the last block."""
    if model.family != "feature_token_transformer":
        return None
    net = model.model
    net.eval()
    X = torch.from_numpy(model._numeric(frame))
    with torch.inference_mode():
        z = net.tokens(X)
        layers = list(net.encoder.layers)
        for layer in layers[:-1]:
            z = layer(z)
        last = layers[-1]
        h = last.norm1(z)
        _, w = last.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
    mean = w[:, 0, 1:].mean(dim=0).numpy()
    return dict(zip(model.columns, mean.astype(float).tolist()))


def run_replicate(units, splits, pairing, replicate, seed, out_root: Path, variants=None, threads=2,
                  importance=False, log=print, min_free_gib=0.6):
    variants = variants or list(VARIANTS)
    frames = pairing_frames(units, splits, pairing, replicate)
    rep_dir = Path(out_root) / pairing / f"rep{replicate}"
    text = None
    with threadpool_limits(limits=threads):
        for variant in variants:
            for weighting in ("none", "balanced"):
                dest = rep_dir / f"{variant}__{weighting}"
                if (dest / "done.json").exists() or not acquire_lock(dest):
                    continue
                wait_for_memory(min_free_gib, log)
                if VARIANTS[variant][0] == "tfidf_logistic" and text is None:
                    text = TextFeatures(frames["fit"])
                proc = psutil.Process()
                rss0 = proc.memory_info().rss
                candidates, best = [], None
                for setting in (0, 1):
                    m = Classifier(variant, setting, weighting, seed, threads=threads, text=text)
                    m.fit(frames["fit"], frames["tune"])
                    candidates.append(dict(setting=setting, params=m.params, tune_ap=m.tune_ap,
                                           fit_seconds=m.fit_seconds, epochs=len(m.history) or None,
                                           best_epoch=getattr(m, "best_epoch", None), n_parameters=m.n_parameters))
                    if best is None or m.tune_ap > best.tune_ap:
                        best = m
                    else:
                        del m
                rows = []
                score_seconds = 0.0
                for key in SCORED:
                    f = frames[key]
                    t0 = time.perf_counter()
                    s = best.score(f)
                    score_seconds += time.perf_counter() - t0
                    rows.append(pd.DataFrame(dict(record_id=f.record_id.to_numpy(), frame=key,
                                                  collection=f.collection.to_numpy(), label=f.label.to_numpy(),
                                                  score=s.astype(np.float64))))
                scores = pd.concat(rows, ignore_index=True)
                if not np.isfinite(scores.score).all():
                    raise FloatingPointError(f"non-finite scores in {dest}")
                dest.mkdir(parents=True, exist_ok=True)
                scores.to_parquet(dest / "scores.parquet", index=False)
                disk = pd.read_parquet(dest / "scores.parquet")
                neg = disk[(disk.frame == "L_calibration") & (disk.label == 0)].score.to_numpy()
                thresholds = [np_threshold(neg, a, DELTA) for a in ALPHAS]
                write_json(dest / "thresholds.json", thresholds)
                text_seconds = text.fit_seconds if VARIANTS[variant][0] == "tfidf_logistic" else 0.0
                n_scored = int(len(scores))
                write_json(dest / "selection.json", dict(
                    pairing=pairing, replicate=replicate, seed=seed, variant=variant, family=best.family,
                    weighting=weighting, columns=best.columns, candidates=candidates, chosen_setting=best.setting,
                    chosen_tune_ap=best.tune_ap,
                    criterion="largest tuning average precision; setting 0 kept on exact ties",
                    selected_fit_seconds=best.fit_seconds + text_seconds,
                    search_fit_seconds=sum(c["fit_seconds"] for c in candidates) + text_seconds,
                    vocabulary_fit_seconds=text_seconds, scoring_seconds=score_seconds,
                    scoring_seconds_per_1000=1000 * score_seconds / n_scored, n_scored=n_scored,
                    rss_before_gib=rss0 / 2**30, rss_after_gib=proc.memory_info().rss / 2**30,
                    peak_working_set_gib=getattr(proc.memory_info(), "peak_wset", 0) / 2**30))
                if best.history:
                    write_json(dest / "history.json", best.history)
                if importance and variant in PERMUTATION_VARIANTS:
                    pop_id = pd.concat([frames["P_test"], frames["L_test"]], ignore_index=True)
                    pop_shift = pd.concat([frames["Pother_all"], frames["Lother_all"]], ignore_index=True)
                    write_json(dest / "permutation_importance.json", dict(
                        definition="mean decrease in average precision over shuffles of one column",
                        in_distribution=permutation_importance(best, pop_id, 5, seed),
                        both_shift=permutation_importance(best, pop_shift, 5, seed)))
                    att = cls_attention(best, pop_id)
                    if att:
                        write_json(dest / "cls_attention.json", dict(
                            definition="mean last-block attention weight from [CLS] to each feature token, "
                                       "in-distribution test units", weights=att))
                write_json(dest / "done.json", dict(utc=datetime.now(timezone.utc).isoformat(),
                                                    available_gib=available_gib()))
                (dest / "lock").unlink(missing_ok=True)
                log(f"{pairing} rep{replicate} {variant} {weighting}: tuneAP={best.tune_ap:.4f} "
                    f"fit={best.fit_seconds:.1f}s avail={available_gib():.2f}GiB")
                del best, scores, disk
                gc.collect()
    del text
    gc.collect()


def environment() -> dict:
    import importlib.metadata as md
    import platform
    import sys
    pk = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "pyarrow", "beautifulsoup4", "langid",
                 "psutil", "threadpoolctl", "matplotlib", "requests", "truststore"]:
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    vm = psutil.virtual_memory()
    return dict(utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                processor=platform.processor(), python=sys.version, logical_processors=os.cpu_count(),
                physical_cores=psutil.cpu_count(logical=False), memory_total_gib=vm.total / 2**30,
                memory_available_gib_at_start=vm.available / 2**30, torch_threads=torch.get_num_threads(),
                device="CPU", packages=pk)
