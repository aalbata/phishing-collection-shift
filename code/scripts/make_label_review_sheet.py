"""Draw the pre-declared label-audit sample: 50 units per collection, ordered by SHA-256(record_id + salt).

The shareable sheet lists hashed identifiers and empty reviewer columns. The private sheet (data/prepared)
adds subject and the first 600 characters of the visible body for human reviewers. No review outcome is
recorded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

SALT = "label-audit-2026-11-05"
PER_COLLECTION = 50


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    a = p.parse_args()
    root = a.project.resolve()
    units = pd.read_parquet(root / "data" / "prepared" / "units.parquet")
    units["order"] = units.record_id.map(lambda r: hashlib.sha256((r + SALT).encode()).hexdigest())
    pick = (units.sort_values("order").groupby("collection", group_keys=False).head(PER_COLLECTION)
            .sort_values(["collection", "order"]))
    cols = ["record_id", "collection", "label"]
    sheet = pick[cols].copy()
    for c in ("reviewer_1_label", "reviewer_2_label", "adjudicated_label", "notes"):
        sheet[c] = ""
    sheet["label_definitions"] = "1 = phishing (credential, payment or account deception); 0 = legitimate; " \
                                 "S = spam or other unsolicited but not phishing; U = cannot decide"
    sheet.to_csv(root / "datasets" / "label_review_sheet.csv", index=False)
    private = pick[cols + ["subject"]].copy()
    private["body_excerpt"] = pick.body.str.slice(0, 600)
    private.to_csv(root / "data" / "prepared" / "label_review_sheet_PRIVATE.csv", index=False)
    print(sheet.groupby("collection").size())


if __name__ == "__main__":
    main()
