"""Build the message table from the retrieved archives and freeze the analysis units.

Steps: parse every candidate message of every available archive; apply the
inclusion rules (decodable, at least 20 non-space characters, identified as
English); link exact, template and near-duplicates across all collections;
quarantine groups whose members carry different collection labels; remove
groups that span more than one collection; keep one representative per
remaining group. Every exclusion is recorded with its reason.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import langid
import numpy as np
import pandas as pd

from .acquire import sha256_file
from .dedup import group_messages
from .features import FEATURES, canonical, engineered, model_text, template_text
from .parse import iter_archive, parse_message, sha256_text
from .sources import COLLECTIONS, archive_specs

MIN_CHARS = 20
ENRON_SAMPLE_MODULUS = 10  # keep a message when int(sha256(member path)[:8], 16) % 10 == 0
ENRON_EXCLUDED_FOLDER_WORDS = ("sent", "deleted", "junk", "spam")
# Automated senders on the Apache lists (issue trackers, CI, repository bots) are not human mail;
# the rule looks at the local part of the From address only.
AUTOMATED_LOCAL = re.compile(r"^(jira|git|gitbox|github|buildbot|jenkins|hudson|builds?|bot|ci|reviews?)$"
                             r"|no-?reply|notification", re.I)


def automated_sender(address: str) -> bool:
    local = (address or "").split("@")[0]
    return bool(local) and bool(AUTOMATED_LOCAL.search(local))


def enron_rule(locator: str, meta: dict) -> str:
    """Return '' when an Enron message is retained, else the rule that removes it."""
    folder = (meta.get("folder") or "").lower()
    if any(w in folder for w in ENRON_EXCLUDED_FOLDER_WORDS):
        return "enron_folder_rule"
    if int(hashlib.sha256(locator.encode()).hexdigest()[:8], 16) % ENRON_SAMPLE_MODULUS:
        return "enron_sampling_rule"
    return ""


def process_message(spec: dict, digest: str, label: int, locator: str, raw: bytes, meta: dict):
    """Apply the inclusion rules to one message; return (row, outcome)."""
    rec = dict(record_id=sha256_text(digest + ":" + locator), collection=spec["collection"],
               archive=spec["name"], archive_sha256=digest, locator=locator,
               locator_sha256=sha256_text(locator), label=label,
               mailbox_user=meta.get("mailbox_user"), folder=meta.get("folder"),
               raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest())
    try:
        p = parse_message(raw)
    except Exception as exc:
        return dict(rec, status="excluded", reason=f"parse_error:{type(exc).__name__}"), "parse_error"
    text = model_text(p["subject"], p["body"])
    rec.update(year_header=p["year"], subject=p["subject"], body=p["body"], text=text)
    if spec["collection"] == "apache_lists" and automated_sender(p["sender"]):
        return dict(rec, status="excluded", reason="automated_sender"), "automated_sender"
    if len(text.replace(" ", "").replace("\n", "")) < MIN_CHARS:
        return dict(rec, status="excluded", reason="too_short"), "too_short"
    lang, score = langid.classify(text[:3000])
    rec.update(language=lang, language_score=float(score))
    rec.update(engineered(p["subject"], p["body"], p))
    if lang != "en":
        return dict(rec, status="excluded", reason="not_english"), "not_english"
    return dict(rec, status="eligible", reason=""), "eligible"


def _zip_worker(args):
    path, spec, digest, label, names = args
    import zipfile
    from pathlib import PurePosixPath
    from .runtime import disable_power_throttling
    disable_power_throttling()
    out = []
    with zipfile.ZipFile(path) as zf:
        for index, name in names:
            raw = zf.read(name)
            locator = "/".join(PurePosixPath(name).parts[1:])
            out.append((index,) + process_message(spec, digest, label, locator, raw, {}))
    return out


def _mbox_worker(args):
    """Parse one whole mbox archive and write its cache files (rows, counts)."""
    spec, path, expected, cache_dir = args
    from .runtime import disable_power_throttling
    disable_power_throttling()
    digest = sha256_file(Path(path))
    if digest != expected:
        raise RuntimeError(f"hash mismatch for {path}")
    label = COLLECTIONS[spec["collection"]]["label"]
    t0, counts, rows = time.time(), Counter(), []
    for locator, raw, meta in iter_archive(Path(path), "mbox", spec["collection"]):
        counts["candidates"] += 1
        row, outcome = process_message(spec, digest, label, locator, raw, meta)
        counts[outcome] += 1
        rows.append(row)
    entry = dict(collection=spec["collection"], archive=spec["name"], available=True, sha256=digest,
                 seconds=round(time.time() - t0, 1), **counts)
    pd.DataFrame(rows).to_parquet(Path(cache_dir) / f"{digest}.parquet", index=False)
    (Path(cache_dir) / f"{digest}.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return f"{spec['collection']}/{spec['name']}", entry


def parse_zip_parallel(path: Path, spec: dict, digest: str, label: int, workers: int):
    """Parse the .eml members of a zip archive in worker processes; rows keep the archive order."""
    import zipfile
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import PurePosixPath
    from .parse import MAX_MESSAGE_BYTES
    with zipfile.ZipFile(path) as zf:
        names = [(i, info.filename) for i, info in enumerate(zf.infolist())
                 if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() == ".eml"
                 and "email" in PurePosixPath(info.filename).parts and info.file_size <= MAX_MESSAGE_BYTES]
    chunks = [names[i::workers * 4] for i in range(workers * 4)]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_zip_worker, [(str(path), spec, digest, label, c) for c in chunks]):
            results.extend(part)
    results.sort(key=lambda r: r[0])
    for _, row, outcome in results:
        yield row, outcome


def parse_collections(raw_root: Path, manifest_path: Path, cache_dir: Path, only=None,
                      log=print, workers: int = 1) -> tuple[pd.DataFrame, list[dict]]:
    """Parse all available archives into one table (private: contains message text).

    Each archive's rows are cached in ``cache_dir`` under its SHA-256, so an unchanged archive is parsed once.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    available = {r["relative_path"]: r for r in manifest["files"]
                 if r.get("kind") != "documentation" and r.get("status") in ("downloaded", "cached", "manual")}
    frames, archive_log = [], []
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    langid.set_languages(None)
    if workers > 1:  # parse uncached mbox archives in parallel, one archive per task
        pending = []
        for spec in archive_specs():
            rel = f"{spec['collection']}/{spec['name']}"
            path = Path(raw_root) / rel
            if (only and spec["collection"] not in only) or spec["kind"] != "mbox" or rel not in available \
                    or not path.is_file():
                continue
            if not (cache_dir / f"{available[rel]['sha256']}.json").exists():
                pending.append((spec, str(path), available[rel]["sha256"], str(cache_dir)))
        if pending:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for rel, entry in pool.map(_mbox_worker, pending):
                    log(f"parsed {rel}: candidates {entry.get('candidates')}, eligible {entry.get('eligible')}")
    for spec in archive_specs():
        if only and spec["collection"] not in only:
            continue
        rel = f"{spec['collection']}/{spec['name']}"
        path = Path(raw_root) / rel
        if rel not in available or not path.is_file():
            archive_log.append(dict(collection=spec["collection"], archive=spec["name"], available=False))
            log(f"unavailable {rel}")
            continue
        digest = sha256_file(path)
        if digest != available[rel]["sha256"]:
            raise RuntimeError(f"hash mismatch for {rel}")
        cached, cached_log = cache_dir / f"{digest}.parquet", cache_dir / f"{digest}.json"
        if cached.exists() and cached_log.exists():
            frames.append(pd.read_parquet(cached))
            archive_log.append(json.loads(cached_log.read_text(encoding="utf-8")))
            log(f"cached parse {rel}")
            continue
        rows = []
        label = COLLECTIONS[spec["collection"]]["label"]
        t0 = time.time()
        counts = Counter()

        def select(locator, meta):
            counts["candidates"] += 1
            rule = enron_rule(locator, meta)
            if rule:
                counts[rule] += 1
            return not rule

        if spec["kind"] == "zip" and workers > 1:
            for row, outcome in parse_zip_parallel(path, spec, digest, label, workers):
                counts["candidates"] += 1
                if outcome is not None:
                    counts[outcome] += 1
                if row is not None:
                    rows.append(row)
        else:
            chooser = select if spec["collection"] == "enron" else None
            for locator, raw, meta in iter_archive(path, spec["kind"], spec["collection"], select=chooser):
                if chooser is None:
                    counts["candidates"] += 1
                if counts["candidates"] % 5000 == 0:
                    log(f"  {rel}: {counts['candidates']} candidates read")
                row, outcome = process_message(spec, digest, label, locator, raw, meta)
                counts[outcome] += 1
                rows.append(row)
        entry = dict(collection=spec["collection"], archive=spec["name"], available=True, sha256=digest,
                     seconds=round(time.time() - t0, 1), **counts)
        archive_log.append(entry)
        part = pd.DataFrame(rows)
        part.to_parquet(cached, index=False)
        cached_log.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        frames.append(part)
        log(f"parsed {rel}: {dict(counts)} in {entry['seconds']}s")
    frames = [f for f in frames if len(f)]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), archive_log


def freeze_units(df: pd.DataFrame, threshold: float = 0.8, log=print) -> tuple[pd.DataFrame, dict]:
    """Group duplicates across collections and keep one representative per single-collection group."""
    df = df.copy()
    df["group_raw"] = None
    df["group_id"] = None
    df["duplicate_of"] = None
    elig = df.index[df.status.eq("eligible")].to_numpy()
    texts = df.loc[elig, "text"].tolist()
    t0 = time.time()
    groups, pairs = group_messages([canonical(t) for t in texts], [template_text(t) for t in texts],
                                   threshold=threshold)
    log(f"grouped {len(elig)} eligible messages into {len(set(groups.tolist()))} groups "
        f"({len(pairs)} confirmed near-duplicate pairs) in {time.time() - t0:.0f}s")
    df.loc[elig, "group_raw"] = [f"g{g:07d}" for g in groups]
    sub = df.loc[elig]
    n_labels = sub.groupby("group_raw")["label"].nunique()
    n_colls = sub.groupby("group_raw")["collection"].nunique()
    conflict = set(n_labels[n_labels > 1].index)
    multi = set(n_colls[n_colls > 1].index) - conflict
    in_conflict = df.group_raw.isin(conflict) & df.status.eq("eligible")
    in_multi = df.group_raw.isin(multi) & df.status.eq("eligible")
    df.loc[in_conflict, ["status", "reason"]] = ["excluded", "label_conflict_group"]
    df.loc[in_multi, ["status", "reason"]] = ["excluded", "cross_collection_group"]
    keep = df.loc[df.status.eq("eligible")].sort_values(["group_raw", "record_id"])
    first = keep.drop_duplicates("group_raw", keep="first")
    dup_idx = keep.index.difference(first.index)
    rep_of = first.set_index("group_raw")["record_id"]
    df.loc[dup_idx, ["status", "reason"]] = ["excluded", "duplicate_within_group"]
    df.loc[dup_idx, "duplicate_of"] = df.loc[dup_idx, "group_raw"].map(rep_of)
    df.loc[first.index, "status"] = "included"
    # Public group identifier: hash of the smallest record id in the group.
    smallest = sub.sort_values("record_id").drop_duplicates("group_raw").set_index("group_raw")["record_id"]
    ghash = {g: hashlib.sha256(("group:" + rid).encode()).hexdigest()[:16] for g, rid in smallest.items()}
    mask = df.group_raw.notna()
    df.loc[mask, "group_id"] = df.loc[mask, "group_raw"].map(ghash)
    sizes = sub.groupby("group_raw").size()
    stats = dict(eligible=int(len(elig)), groups=int(len(sizes)), near_duplicate_pairs=len(pairs),
                 largest_group=int(sizes.max()) if len(sizes) else 0,
                 label_conflict_groups=len(conflict), label_conflict_messages=int(in_conflict.sum()),
                 cross_collection_groups=len(multi), cross_collection_messages=int(in_multi.sum()),
                 included=int(df.status.eq("included").sum()),
                 exclusion_reasons={k: int(v) for k, v in
                                    Counter(df.loc[df.status.eq("excluded"), "reason"]).items()},
                 near_duplicate_similarity_quantiles=(np.quantile([p[2] for p in pairs], [0, .5, 1]).round(4).tolist()
                                                      if pairs else []),
                 threshold=threshold, created_utc=datetime.now(timezone.utc).isoformat())
    return df, stats


PUBLIC_COLUMNS = ["record_id", "collection", "archive", "archive_sha256", "locator_sha256", "raw_sha256",
                  "label", "year_header", "language", "status", "reason", "group_id"] + FEATURES
