"""Reproduce the study.

Fast path (default): from the shipped evidence (saved scores, frozen splits and unit manifest), recompute every
metric, the declared tests, the independent audit, the figures and the report. No mail archive is needed.

Full path (--full): retrieve the original archives, rebuild the corpus, refit every model, then run the fast path.
The full path needs internet access and several hours on a laptop CPU.

The undeclared check of the confirmatory decisions under eleven further bootstrap seeds runs with --seed-check
(always on the full path); it takes about an hour with six worker processes.

Usage: python -I code/scripts/reproduce.py [--full] [--skip-bootstrap-audit] [--seed-check]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
SCRIPTS = CODE / "scripts"


TIMES: dict[str, float] = {}


def step(name, args):
    t0 = time.time()
    print(f"== {name}", flush=True)
    r = subprocess.run([sys.executable, "-I", str(SCRIPTS / args[0]), *args[1:]])
    if r.returncode:
        raise SystemExit(f"{name} failed with exit code {r.returncode}")
    TIMES[name] = time.time() - t0
    print(f"   done in {TIMES[name]:.0f} s", flush=True)


def unpack_probes(root: Path) -> None:
    import shutil
    src, dst = root / "evidence" / "packed" / "probes", root / "evidence" / "probes"
    if not dst.exists():
        shutil.copytree(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true")
    p.add_argument("--skip-bootstrap-audit", action="store_true",
                   help="audit everything except the 2,000-draw bootstrap recomputation")
    p.add_argument("--seed-check", action="store_true",
                   help="repeat the confirmatory bootstrap with eleven further seeds (bootstrap_sensitivity.py)")
    a = p.parse_args()
    start = time.time()
    if a.full:
        step("retrieve archives", ["acquire_data.py"])
        step("check Phishing Pot against the Git tree of its commit", ["check_phishing_pot.py"])
        step("build corpus and splits", ["build_corpus.py"])
        step("fit the grid", ["run_experiments.py"])
        step("collection-identity probes", ["run_probes.py"])
    else:
        root = CODE.parent
        if not any((root / "evidence" / "runs").glob("*/rep*/*__*/done.json")):
            sys.path.insert(0, str(CODE))
            from phishstudy.evidence_pack import unpack
            print("== restore saved scores from evidence/packed", flush=True)
            n = unpack(root / "evidence" / "packed", root / "evidence" / "runs")
            print(f"   {n} fits restored", flush=True)
            if (root / "evidence" / "packed" / "probes").exists():
                unpack_probes(root)
    step("metrics and declared tests", ["analyze.py"])
    step("independent audit", ["verify_all.py"] + (["--skip-bootstrap"] if a.skip_bootstrap_audit else []))
    if a.full or a.seed_check:
        step("confirmatory decisions under further bootstrap seeds", ["bootstrap_sensitivity.py"])
    step("figures", ["make_figures.py"])
    step("report", ["make_report.py"])
    import json
    import platform
    out = CODE.parent / "evidence" / "logs" / "reproduce_timing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(path="full" if a.full else "fast", steps_seconds=TIMES,
                                   total_seconds=time.time() - start, python=platform.python_version(),
                                   processor=platform.processor()), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
