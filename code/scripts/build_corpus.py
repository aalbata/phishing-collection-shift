"""Parse the retrieved archives, freeze the analysis units and the replicate splits.

Writes (private, contains message text): data/prepared/messages.parquet, data/prepared/units.parquet
Writes (shareable, no text): datasets/corpus_manifest.csv.gz, datasets/splits.csv.gz,
datasets/corpus_summary.json, datasets/archive_log.json
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

import pandas as pd  # noqa: E402

from phishstudy.corpus import PUBLIC_COLUMNS, freeze_units, parse_collections  # noqa: E402
from phishstudy.splits import assign, split_hash  # noqa: E402
from phishstudy.acquire import sha256_file  # noqa: E402

SEEDS = [17, 29, 43, 71, 101]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    p.add_argument("--parse-only", nargs="*", default=None,
                   help="parse (and cache) these collections only, without freezing units")
    p.add_argument("--workers", type=int, default=4, help="worker processes for zip archives")
    a = p.parse_args()
    root = a.project.resolve()
    prepared = root / "data" / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    cache = prepared / "parsed_archives"
    if a.parse_only is not None:
        parse_collections(root / "data" / "raw", root / "datasets" / "download_manifest.json", cache,
                          only=a.parse_only, workers=a.workers)
        return
    df, archive_log = parse_collections(root / "data" / "raw", root / "datasets" / "download_manifest.json", cache,
                                        workers=a.workers)
    df.to_parquet(prepared / "messages.parquet", index=False)
    (root / "datasets" / "archive_log.json").write_text(json.dumps(archive_log, indent=2), encoding="utf-8")
    df, stats = freeze_units(df)
    units = df[df.status.eq("included")].reset_index(drop=True)
    units.to_parquet(prepared / "units.parquet", index=False)
    df[[c for c in PUBLIC_COLUMNS + ["duplicate_of"] if c in df.columns]].to_csv(
        root / "datasets" / "corpus_manifest.csv.gz", index=False)
    splits = assign(units, SEEDS)
    splits.to_csv(root / "datasets" / "splits.csv.gz", index=False)
    by_coll = (df.groupby(["collection", "status", "reason"], dropna=False).size()
               .rename("messages").reset_index())
    stats.update(
        seeds=SEEDS, split_sha256=split_hash(splits),
        units_sha256=sha256_file(prepared / "units.parquet"),
        units_per_collection=units.collection.value_counts().sort_index().to_dict(),
        status_by_collection=by_coll.to_dict("records"),
        units_years=units.groupby("collection").year_header.describe().round(1).to_dict("index"))
    (root / "datasets" / "corpus_summary.json").write_text(json.dumps(stats, indent=2, default=str),
                                                         encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("eligible", "groups", "included", "units_per_collection",
                                             "label_conflict_groups", "cross_collection_groups")}, indent=2))


if __name__ == "__main__":
    main()
