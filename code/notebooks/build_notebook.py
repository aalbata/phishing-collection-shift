"""Write Phishing_Research_End_to_End.ipynb (cells call the package; nothing is computed here)."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# Phishing email detection across public mail collections

This notebook runs the complete study on the real archives: retrieval records, corpus freeze checks, software
tests, the resumable experiment grid, metrics, the independent numerical verification and the figures. It uses
only the original public archives listed in `datasets/DATASET_PROVENANCE.csv`; no message is generated, oversampled
or relabelled. Message text never appears in the outputs.

Run order: top to bottom. On a fresh machine, run `python -I code/scripts/acquire_data.py` and
`python -I code/scripts/build_corpus.py` first (about one hour on a laptop CPU), or point `PROJECT` at a copy that
already holds `data/`. Completed fits are detected and reused, so a second run only recomputes metrics.
""")
code("""
import json, os, platform, subprocess, sys, time
from pathlib import Path
PROJECT = Path(os.environ.get("PHISH_PROJECT", Path.cwd().parents[1] if Path.cwd().name == "notebooks" else Path.cwd()))
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
import pandas as pd
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
from phishstudy.runtime import disable_power_throttling
disable_power_throttling()
from phishstudy.experiment import environment
env = environment()
print(json.dumps({k: env[k] for k in ["platform", "processor", "python", "logical_processors", "memory_total_gib"]}, indent=1))
print(json.dumps(env["packages"], indent=1))
""")
md("## 1. Retrieved archives\nEach archive was fetched from its original host; the SHA-256 digest identifies the snapshot. "
   "Files that could not be retrieved (datasets/BLOCKED_ARCHIVES.md) are listed as failed and were not replaced.")
code("""
manifest = json.loads((PROJECT / "datasets" / "download_manifest.json").read_text(encoding="utf-8"))
m = pd.DataFrame(manifest["files"])
m = m[m.kind != "documentation"]
view = m[["collection", "name", "status", "bytes"]].copy()
view["sha256"] = m["sha256"].fillna("").str.slice(0, 16)
print(view.groupby(["collection", "status"]).size())
view
""")
md("## 2. Frozen corpus\nThe hashes below must equal those recorded when the units and splits were frozen.")
code("""
from phishstudy.acquire import sha256_file
from phishstudy.splits import split_hash
summary = json.loads((PROJECT / "datasets" / "corpus_summary.json").read_text(encoding="utf-8"))
splits = pd.read_csv(PROJECT / "datasets" / "splits.csv.gz")
PRIVATE = (PROJECT / "data" / "prepared" / "units.parquet").exists()
assert split_hash(splits) == summary["split_sha256"]
if PRIVATE:
    assert sha256_file(PROJECT / "data" / "prepared" / "units.parquet") == summary["units_sha256"]
    print("private units and splits match their frozen hashes")
else:
    print("splits match their frozen hash; private corpus not present, saved evidence is used")
manifest_units = pd.read_csv(PROJECT / "datasets" / "corpus_manifest.csv.gz")
counts = manifest_units.pivot_table(index="collection", columns="status", values="record_id", aggfunc="count", fill_value=0)
counts
""")
code("""
reasons = manifest_units[manifest_units.status.eq("excluded")].groupby(["collection", "reason"]).size().unstack(fill_value=0)
reasons
""")
md("## 3. Software and integration tests")
code("""
r = subprocess.run([sys.executable, "-I", "-m", "pytest", str(CODE / "tests"), "-q", "-p", "no:cacheprovider"],
                   capture_output=True, text=True, cwd=str(CODE))
print(r.stdout[-2000:]); print(r.stderr[-2000:])
assert r.returncode == 0
""")
md("## 4. Experiment grid\nFour training pairings, five replicates, ten variants, two class weightings. "
   "Completed fits are detected and skipped; missing fits are run.")
code("""
t0 = time.time()
if PRIVATE:
    r = subprocess.run([sys.executable, "-I", str(CODE / "scripts" / "run_experiments.py")], capture_output=True, text=True)
    print(r.stdout[-1500:]); print(r.stderr[-1500:])
    assert r.returncode == 0
elif not any((PROJECT / "evidence" / "runs").glob("*/rep*/*__*/done.json")):
    from phishstudy.evidence_pack import unpack
    print(unpack(PROJECT / "evidence" / "packed", PROJECT / "evidence" / "runs"), "fits restored from the packed evidence")
runs = sorted((PROJECT / "evidence" / "runs").glob("*/rep*/*__*/done.json"))
print(len(runs), "completed fits", f"({time.time() - t0:.0f} s)")
""")
md("Determinism check: one replicate of the feature-token Transformer on the primary pairing is refitted in a "
   "scratch directory and its saved scores are compared with the evidence files.")
code("""
import tempfile, numpy as np
from phishstudy.experiment import run_replicate
from phishstudy.features import FEATURES


def determinism_check():
    units = pd.read_parquet(PROJECT / "data" / "prepared" / "units.parquet",
                            columns=["record_id", "collection", "label", "text"] + FEATURES)
    seed = int(splits[splits.replicate.eq(0)].seed.iloc[0])
    with tempfile.TemporaryDirectory() as tmp:
        run_replicate(units, splits, "NZ-SA", 0, seed, Path(tmp), variants=["feature_token_transformer"],
                      threads=3, importance=True, log=lambda *a: None)
        for w in ("none", "balanced"):
            new = pd.read_parquet(Path(tmp) / "NZ-SA" / "rep0" / f"feature_token_transformer__{w}" / "scores.parquet")
            old = pd.read_parquet(PROJECT / "evidence" / "runs" / "NZ-SA" / "rep0" /
                                  f"feature_token_transformer__{w}" / "scores.parquet")
            diff = float(np.max(np.abs(new.score.to_numpy() - old.score.to_numpy())))
            print(w, "max absolute score difference", diff)
            assert (new.record_id.to_numpy() == old.record_id.to_numpy()).all() and diff < 1e-9
            for name in ("permutation_importance.json", "cls_attention.json"):
                a = json.loads((Path(tmp) / "NZ-SA" / "rep0" / f"feature_token_transformer__{w}" / name).read_text())
                b = json.loads((PROJECT / "evidence" / "runs" / "NZ-SA" / "rep0" / f"feature_token_transformer__{w}" /
                                name).read_text())
                flat_a, flat_b = pd.json_normalize(a).iloc[0], pd.json_normalize(b).iloc[0]
                assert list(flat_a.index) == list(flat_b.index)
                num = [k for k in flat_a.index if isinstance(flat_a[k], float)]
                gap = max(abs(flat_a[k] - flat_b[k]) for k in num)
                print(w, name, "values compared:", len(num), "max absolute difference", gap)
                assert gap < 1e-9


if PRIVATE:
    determinism_check()
else:
    print("private corpus not present: refitting is part of the full path (scripts/reproduce.py --full)")
""")
md("## 5. Metrics, declared tests and descriptive analyses")
code("""
r = subprocess.run([sys.executable, "-I", str(CODE / "scripts" / "analyze.py")], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-1500:])
assert r.returncode == 0
""")
code("""
M = PROJECT / "evidence" / "metrics"
metrics = pd.read_parquet(M / "run_metrics.parquet")
sel = metrics[metrics.selected & metrics.alpha.isna()]
table = sel.groupby(["pairing", "variant", "population"]).average_precision.mean().unstack("population").round(3)
table
""")
code("""
pd.read_csv(M / "confirmatory_tests.csv")[["population", "model_a", "model_b", "delta", "ci_low", "ci_high", "p_value", "holm_p"]].round(4)
""")
code("""
op = metrics[metrics.selected & metrics.alpha.eq(0.01)]
op.groupby(["pairing", "variant", "population"])[["recall", "fpr"]].mean().round(4).unstack("population")
""")
code("""
pd.read_csv(M / "probe_metrics.csv").groupby(["probe", "variant"])[["average_precision", "roc_auc"]].agg(["mean", "std"]).round(4)
""")
md("## 6. Independent verification\nA separate module recomputes every stored number from the saved scores and stops on any mismatch.")
code("""
r = subprocess.run([sys.executable, "-I", str(CODE / "scripts" / "verify_all.py")], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-1500:])
assert r.returncode == 0
""")
md("## 7. Figures")
code("""
r = subprocess.run([sys.executable, "-I", str(CODE / "scripts" / "make_figures.py")], capture_output=True, text=True)
print(r.stdout[-1500:]); print(r.stderr[-1500:])
assert r.returncode == 0
from IPython.display import Image, display
for png in sorted((PROJECT / "evidence" / "figures").glob("Figure_*.png")):
    print(png.name); display(Image(filename=str(png), width=760))
""")
md("""
## 8. Scope
The collections were labelled by their collectors; labels have not been reviewed by the authors (a pre-declared
review sheet is in `datasets/label_review_sheet.csv`). Results describe these public collections, not a deployed
mail service, and the realised false-positive rates under collection shift are measurements, not guarantees.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).with_name("Phishing_Research_End_to_End.ipynb")
nbf.write(nb, out)
print("written", out, len(cells), "cells")
