"""Integration checks on the real frozen corpus (skipped when the private data are absent)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from phishstudy.acquire import sha256_file  # noqa: E402
from phishstudy.experiment import PAIRINGS, pairing_frames  # noqa: E402
from phishstudy.features import FEATURES  # noqa: E402
from phishstudy.sources import COLLECTIONS  # noqa: E402
from phishstudy.splits import split_hash  # noqa: E402

UNITS = ROOT / "data" / "prepared" / "units.parquet"
pytestmark = pytest.mark.skipif(not UNITS.exists(), reason="private corpus not built")


@pytest.fixture(scope="module")
def corpus():
    units = pd.read_parquet(UNITS)
    splits = pd.read_csv(ROOT / "datasets" / "splits.csv.gz")
    summary = json.loads((ROOT / "datasets" / "corpus_summary.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(ROOT / "datasets" / "corpus_manifest.csv.gz")
    return units, splits, summary, manifest


def test_frozen_hashes(corpus):
    units, splits, summary, _ = corpus
    assert sha256_file(UNITS) == summary["units_sha256"]
    assert split_hash(splits) == summary["split_sha256"]


def test_units_are_unique_english_single_collection_groups(corpus):
    units, _, _, manifest = corpus
    assert units.record_id.is_unique and units.group_id.is_unique
    assert set(units.language) == {"en"}
    assert (units.text.str.replace(r"\s", "", regex=True).str.len() >= 20).all()
    elig = manifest[manifest.group_id.notna()]
    spans = elig.groupby("group_id").collection.nunique()
    included_groups = set(units.group_id)
    assert (spans.loc[list(included_groups)] == 1).all()


def test_labels_follow_collections(corpus):
    units = corpus[0]
    for coll, meta in COLLECTIONS.items():
        part = units[units.collection.eq(coll)]
        if len(part):
            assert set(part.label) == {meta["label"]}


def test_features_finite_and_nonnegative(corpus):
    units = corpus[0]
    X = units[FEATURES]
    assert X.notna().all().all() and (X >= 0).all().all()


def test_splits_partition_every_unit(corpus):
    units, splits, _, _ = corpus
    for rep, part in splits.groupby("replicate"):
        assert part.record_id.is_unique
        assert set(part.record_id) == set(units.record_id)


def test_pairing_frames_are_disjoint(corpus):
    units, splits, _, _ = corpus
    present = set(units.collection)
    for pairing, (p, l) in PAIRINGS.items():
        if p not in present or l not in present:
            continue
        fr = pairing_frames(units, splits, pairing, 0)
        ids = [set(f.record_id) for f in fr.values()]
        assert sum(map(len, ids)) == len(set().union(*ids))
        assert set(fr["fit"].collection) == {p, l}


def test_manifest_has_no_text_columns(corpus):
    manifest = corpus[3]
    assert not {"subject", "body", "text", "locator"} & set(manifest.columns)
