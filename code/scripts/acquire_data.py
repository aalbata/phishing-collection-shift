"""Retrieve the original archives listed in phishstudy/sources.py.

Usage: python -I code/scripts/acquire_data.py [--project ROOT] [--only COLLECTION ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from phishstudy.acquire import acquire, check_format, sha256_file  # noqa: E402
from phishstudy.sources import archive_specs  # noqa: E402


def register_local(root: Path) -> None:
    """Hash manually supplied copies of registered archives and add them to the manifest."""
    manifest_path = root / "datasets" / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {r["relative_path"]: r for r in manifest["files"]}
    for spec in archive_specs():
        rel = f"{spec['collection']}/{spec['name']}"
        path = root / "data" / "raw" / rel
        if not path.is_file() or files.get(rel, {}).get("status") in ("downloaded", "cached"):
            continue
        check_format(path, spec["kind"], spec.get("allow_empty", False))
        rec = dict(spec, relative_path=rel, status="manual", bytes=path.stat().st_size, sha256=sha256_file(path),
                   registered_utc=datetime.now(timezone.utc).isoformat(),
                   note="copied manually from the original URL on an unfiltered network")
        files[rel] = rec
        path.with_name(path.name + ".retrieval.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print("registered", rel, rec["sha256"])
    manifest["files"] = sorted(files.values(), key=lambda r: r["relative_path"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--register-local", action="store_true",
                   help="record archives that were copied manually into data/raw/<collection>/ "
                        "(for files that could not be downloaded); each is format-checked and hashed")
    a = p.parse_args()
    root = a.project.resolve()
    if a.register_local:
        register_local(root)
        return
    m = acquire(root / "data" / "raw", root / "datasets" / "documentation",
                root / "datasets" / "download_manifest.json", include=a.only)
    print("complete:", m["complete"])


if __name__ == "__main__":
    main()
