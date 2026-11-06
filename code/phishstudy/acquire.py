"""Download the original archives over HTTPS with integrity records.

Only the fixed URLs in ``sources.py`` are requested. Links inside messages are
never visited. Each file is written to a ``.part`` file, checked for size and
format, hashed with SHA-256 and only then moved into place. A JSON sidecar and a
manifest record the retrieval.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

try:  # verify certificates against the operating-system trust store when available
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
import requests

from .sources import archive_specs, documentation_specs

USER_AGENT = "phishstudy-archive-retrieval/1.0 (academic research; contact via project page)"
MAGIC = {"mbox": [b"From "], "zip": [b"PK\x03\x04"], "tar": [b"BZh", b"\x1f\x8b"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_format(path: Path, kind: str, allow_empty: bool = False) -> None:
    if path.stat().st_size == 0:
        if allow_empty:  # a list month without messages
            return
        raise ValueError("empty download")
    with path.open("rb") as stream:
        head = stream.read(4096)
    if kind == "documentation":
        return
    if head.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError("received an HTML page instead of an archive")
    if not any(head.lstrip().startswith(m) for m in MAGIC[kind]):
        raise ValueError(f"unexpected leading bytes for a {kind} archive: {head[:8]!r}")


def _fetch(spec: dict, dest: Path, attempts: int = 4, timeout=(20, 120)) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            headers = {"User-Agent": USER_AGENT}
            offset = part.stat().st_size if part.exists() else 0
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(spec["url"], stream=True, timeout=timeout, headers=headers,
                              allow_redirects=True) as r:
                if not r.url.startswith("https://"):
                    raise ValueError(f"redirected away from HTTPS: {r.url}")
                if offset and r.status_code != 206:
                    offset = 0  # server ignored or refused the range request; restart
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                mode = "ab" if offset else "wb"
                total = offset
                deadline = time.monotonic() + spec.get("max_seconds", 3600)
                with part.open(mode) as out:
                    for block in r.iter_content(2**20):
                        total += len(block)
                        if total > spec["max_bytes"]:
                            raise ValueError("file exceeded the configured size limit")
                        if time.monotonic() > deadline:
                            raise TimeoutError("transfer exceeded the per-file time limit")
                        out.write(block)
                declared = r.headers.get("Content-Length")
                encoded = r.headers.get("Content-Encoding")
                # Content-Length counts transfer bytes; it is comparable only without content coding.
                if declared is not None and not encoded and r.status_code == 200 and total != int(declared):
                    raise ValueError(f"size {total} differs from Content-Length {declared}")
                meta = dict(final_url=r.url, http_status=r.status_code,
                            content_type=r.headers.get("Content-Type"), content_encoding=encoded,
                            content_length=declared, last_modified=r.headers.get("Last-Modified"),
                            etag=r.headers.get("ETag"))
            check_format(part, spec["kind"], spec.get("allow_empty", False))
            part.replace(dest)
            return dict(status="downloaded", attempts=attempt, errors=errors, **meta)
        except Exception as exc:  # recorded, retried, never replaced by other data
            message = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            raw_root = str(dest.parent.parent)
            for form in (raw_root.replace("\\", "\\\\"), raw_root):  # keep local folder names out of the record
                message = message.replace(form, "<raw>")
            errors.append(message[:400])
            interrupted = isinstance(exc, (requests.exceptions.ChunkedEncodingError, TimeoutError,
                                           requests.exceptions.ConnectionError))
            if not interrupted:  # only an interrupted transfer may be resumed
                part.unlink(missing_ok=True)
            time.sleep(min(30, 3 * attempt))
    part.unlink(missing_ok=True)
    return dict(status="failed", attempts=attempts, errors=errors)


def acquire(raw_root: Path, docs_root: Path, manifest_path: Path, include=None) -> dict:
    """Retrieve every registered archive and documentation page; never substitute data."""
    raw_root, docs_root = Path(raw_root), Path(docs_root)
    records = []
    for spec in archive_specs() + documentation_specs():
        if include and spec["collection"] not in include:
            continue
        root = docs_root if spec["kind"] == "documentation" else raw_root
        dest = root / spec["collection"] / spec["name"]
        sidecar = dest.with_name(dest.name + ".retrieval.json")
        if dest.exists() and sidecar.exists():
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            if cached.get("sha256") == sha256_file(dest) and cached.get("url") == spec["url"]:
                records.append(dict(cached, status="cached"))
                print(f"cached      {spec['collection']}/{spec['name']}", flush=True)
                continue
        started = datetime.now(timezone.utc).isoformat()
        outcome = _fetch(spec, dest)
        record = dict(spec, relative_path=f"{spec['collection']}/{spec['name']}",
                      retrieval_started_utc=started,
                      retrieved_at_utc=datetime.now(timezone.utc).isoformat(), **outcome)
        if outcome["status"] == "downloaded":
            record.update(bytes=dest.stat().st_size, sha256=sha256_file(dest))
            sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
        print(f"{record['status']:<11} {spec['collection']}/{spec['name']} {record.get('bytes', '')}", flush=True)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), files=records,
                    complete=all(r["status"] in ("downloaded", "cached") for r in records),
                    note="SHA-256 identifies the retrieved snapshot; it does not authenticate labels.")
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Earlier records (including failures) stay unless this run attempted the same file again.
        keep = {r["relative_path"]: r for r in old.get("files", []) if r.get("relative_path")}
        for r in records:
            keep[r.get("relative_path", r["collection"] + "/" + r["name"])] = r
        manifest["files"] = sorted(keep.values(), key=lambda r: r.get("relative_path", ""))
        manifest["complete"] = all(r["status"] in ("downloaded", "cached") for r in manifest["files"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
