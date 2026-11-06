"""Collection-identity probes on the frozen units and splits (resumable)."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from phishstudy.runtime import disable_power_throttling  # noqa: E402

disable_power_throttling()

import pandas as pd  # noqa: E402

from phishstudy.features import FEATURES  # noqa: E402

from phishstudy.probe import run_probe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    root = a.project.resolve()
    units = pd.read_parquet(root / "data" / "prepared" / "units.parquet",
                            columns=["record_id", "collection", "label", "text"] + FEATURES)
    splits = pd.read_csv(root / "datasets" / "splits.csv.gz")
    seeds = splits.drop_duplicates("replicate").set_index("replicate")["seed"].to_dict()
    log_path = root / "evidence" / "logs" / "probe_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    df = run_probe(units, splits, seeds, root / "evidence" / "probes", threads=a.threads, log=log)
    out = root / "evidence" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "probe_metrics.csv", index=False)
    print(df.groupby(["probe", "variant"])[["average_precision", "roc_auc"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
