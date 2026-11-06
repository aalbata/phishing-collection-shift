"""Frozen, group-disjoint splits of the analysis units.

Every unit is the single representative of one duplicate group, so splitting
units keeps groups intact. Within each collection and replicate the units are
shuffled with a seed derived from the replicate seed and the collection name,
then cut into fit, tune, calibration and test parts (50/10/20/20).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

FRACTIONS = (("fit", 0.5), ("tune", 0.1), ("calibration", 0.2), ("test", 0.2))


def collection_seed(seed: int, collection: str) -> int:
    return int(hashlib.sha256(f"{seed}:{collection}".encode()).hexdigest()[:8], 16)


def assign(units: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for r, seed in enumerate(seeds):
        for coll, part in units.groupby("collection", sort=True):
            ids = np.sort(part.record_id.to_numpy())
            perm = np.random.default_rng(collection_seed(seed, coll)).permutation(len(ids))
            ids = ids[perm]
            cuts = np.floor(np.cumsum([f for _, f in FRACTIONS]) * len(ids)).astype(int)
            cuts[-1] = len(ids)
            start = 0
            for (name, _), stop in zip(FRACTIONS, cuts):
                rows.append(pd.DataFrame(dict(record_id=ids[start:stop], collection=coll, replicate=r,
                                              seed=seed, split=name)))
                start = stop
    out = pd.concat(rows, ignore_index=True)
    assert not out.duplicated(["record_id", "replicate"]).any()
    return out


def split_hash(splits: pd.DataFrame) -> str:
    s = splits.sort_values(["replicate", "record_id"])[["replicate", "record_id", "split"]]
    return hashlib.sha256(s.to_csv(index=False).encode()).hexdigest()
