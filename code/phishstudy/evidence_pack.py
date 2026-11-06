"""Pack and unpack the saved scores of every fit into one compact file per training pairing.

Packed form: unit index (position of the record identifier in the sorted unit list), frame code and the float64
score, plus the small JSON files of each fit. Unpacking restores scores.parquet files that are equal row for row and
value for value to the originals (checked by ``verify_roundtrip``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FRAMES = ["P_calibration", "L_calibration", "P_test", "L_test", "Pother_all", "Lother_all", "Mod_all"]
SMALL = ["selection.json", "thresholds.json", "done.json", "history.json", "permutation_importance.json",
         "cls_attention.json"]


def unit_index(manifest: pd.DataFrame) -> pd.DataFrame:
    u = manifest[manifest.status.eq("included")][["record_id", "collection", "label"]]
    return u.sort_values("record_id").reset_index(drop=True)


def pack(runs: Path, manifest: pd.DataFrame, out_dir: Path) -> list[Path]:
    units = unit_index(manifest)
    pos = pd.Series(np.arange(len(units), dtype=np.int32), index=units.record_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for pairing_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        parts, small = [], {}
        for done in sorted(pairing_dir.glob("rep*/*__*/done.json")):
            d = done.parent
            rep = int(d.parent.name[3:])
            variant, weighting = d.name.split("__")
            sc = pd.read_parquet(d / "scores.parquet")
            parts.append(pd.DataFrame(dict(
                replicate=np.int8(rep), variant=variant, weighting=weighting,
                frame=pd.Categorical(sc.frame, categories=FRAMES), unit=pos.loc[sc.record_id].to_numpy(),
                score=sc.score.to_numpy(dtype=np.float64), row=np.arange(len(sc), dtype=np.int32))))
            small[f"rep{rep}/{d.name}"] = {f: json.loads((d / f).read_text(encoding="utf-8"))
                                          for f in SMALL if (d / f).exists()}
        table = pd.concat(parts, ignore_index=True)
        for c in ("variant", "weighting"):
            table[c] = table[c].astype("category")
        target = out_dir / f"scores_{pairing_dir.name}.parquet"
        table.to_parquet(target, index=False, compression="zstd", compression_level=19)
        (out_dir / f"fits_{pairing_dir.name}.json").write_text(json.dumps(small), encoding="utf-8")
        written += [target, out_dir / f"fits_{pairing_dir.name}.json"]
    units.to_csv(out_dir / "unit_index.csv.gz", index=False)
    return written


def unpack(packed: Path, runs: Path) -> int:
    units = pd.read_csv(packed / "unit_index.csv.gz")
    n = 0
    for f in sorted(packed.glob("scores_*.parquet")):
        pairing = f.stem.split("_", 1)[1]
        table = pd.read_parquet(f)
        small = json.loads((packed / f"fits_{pairing}.json").read_text(encoding="utf-8"))
        for (rep, variant, weighting), g in table.groupby(["replicate", "variant", "weighting"], observed=True):
            d = runs / pairing / f"rep{rep}" / f"{variant}__{weighting}"
            d.mkdir(parents=True, exist_ok=True)
            g = g.sort_values("row")
            u = units.iloc[g.unit.to_numpy()]
            pd.DataFrame(dict(record_id=u.record_id.to_numpy(), frame=g.frame.astype(str).to_numpy(),
                              collection=u.collection.to_numpy(), label=u.label.to_numpy(),
                              score=g.score.to_numpy())).to_parquet(d / "scores.parquet", index=False)
            for name, obj in small[f"rep{rep}/{d.name}"].items():
                (d / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
            n += 1
    return n


def verify_roundtrip(original: Path, restored: Path) -> int:
    checked = 0
    for done in sorted(original.glob("*/rep*/*__*/done.json")):
        rel = done.parent.relative_to(original)
        a = pd.read_parquet(done.parent / "scores.parquet")
        b = pd.read_parquet(restored / rel / "scores.parquet")
        if not (a.record_id.equals(b.record_id) and a.frame.equals(b.frame) and a.label.equals(b.label)
                and np.array_equal(a.score.to_numpy(), b.score.to_numpy())):
            raise AssertionError(f"packed scores differ for {rel}")
        checked += 1
    return checked
