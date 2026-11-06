"""Run the declared grid: every pairing, replicate, variant and weighting (resumable).

Usage: python -I code/scripts/run_experiments.py [--pairings NZ-SA ...] [--replicates 0 1 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from phishstudy.runtime import disable_power_throttling  # noqa: E402

disable_power_throttling()

import pandas as pd  # noqa: E402

from phishstudy.features import FEATURES  # noqa: E402

from phishstudy.experiment import PAIRINGS, environment, run_replicate, write_json  # noqa: E402
from phishstudy.models import VARIANTS  # noqa: E402

PRIMARY = "NZ-SA"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    p.add_argument("--pairings", nargs="*", default=list(PAIRINGS))
    p.add_argument("--replicates", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    root = a.project.resolve()
    out = root / "evidence" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    units = pd.read_parquet(root / "data" / "prepared" / "units.parquet",
                            columns=["record_id", "collection", "label", "text"] + FEATURES)
    splits = pd.read_csv(root / "datasets" / "splits.csv.gz")
    seeds = splits.drop_duplicates("replicate").set_index("replicate")["seed"].to_dict()
    log_path = root / "evidence" / "logs" / "run_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    env_path = root / "evidence" / "environment.json"
    if not env_path.exists():
        write_json(env_path, environment())
    log(f"start pairings={a.pairings} replicates={a.replicates} variants={len(a.variants)} threads={a.threads}")
    t0 = time.time()
    for pairing in a.pairings:
        for rep in a.replicates:
            try:
                run_replicate(units, splits, pairing, rep, int(seeds[rep]), out, variants=a.variants,
                              threads=a.threads, importance=(pairing == PRIMARY), log=log)
            except Exception as exc:  # recorded; the grid is re-run, never silently shortened
                log(f"FAILED {pairing} rep{rep}: {type(exc).__name__}: {exc}")
                raise
    log(f"finished in {(time.time() - t0) / 60:.1f} min")
    status = dict(utc=datetime.now(timezone.utc).isoformat(), pairings=a.pairings, replicates=a.replicates,
                  variants=a.variants, completed_runs=len(list(out.glob("*/rep*/*__*/done.json"))))
    (root / "evidence" / "logs" / "grid_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
