"""Check the Phishing Pot archive file by file against the Git tree of the pinned commit.

Every file of the downloaded repository archive is hashed as a Git blob (SHA-1 of "blob <size>\\0" followed by the
content) and compared with the blob identifier that the GitHub tree record of the pinned commit lists for the same
path. Writes datasets/phishing_pot_git_check.json. File contents are hashed only; nothing is parsed or opened.

Usage: python -I code/scripts/check_phishing_pot.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from phishstudy.sources import PHISHING_POT_COMMIT  # noqa: E402


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    root = p.parse_args().project.resolve()
    tree = json.loads((root / "datasets" / "documentation" / "phishing_pot" /
                       f"phishing_pot_git_tree_{PHISHING_POT_COMMIT[:8]}.json").read_text(encoding="utf-8"))
    if tree["sha"] != PHISHING_POT_COMMIT or tree.get("truncated"):
        raise SystemExit("tree record does not belong to the pinned commit or is truncated")
    expected = {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}
    archive = next((root / "data" / "raw" / "phishing_pot").glob(f"phishing_pot_{PHISHING_POT_COMMIT[:12]}.zip"))
    seen, mismatched, missing = 0, [], []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            path = info.filename.split("/", 1)[1]  # strip the archive's top-level folder
            if path not in expected:
                missing.append(path)
                continue
            seen += 1
            if git_blob_sha1(z.read(info)) != expected[path]:
                mismatched.append(path)
    absent = sorted(set(expected) - set(info.filename.split("/", 1)[1] for info in zipfile.ZipFile(archive).infolist()
                                        if not info.is_dir()))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), commit=PHISHING_POT_COMMIT,
                  archive=archive.name, tree_blobs=len(expected), archive_files_checked=seen,
                  identical=seen - len(mismatched), mismatched=mismatched, not_in_tree=missing,
                  in_tree_not_in_archive=absent,
                  status="PASS" if not mismatched and not missing and not absent else "FAIL")
    (root / "datasets" / "phishing_pot_git_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("mismatched", "not_in_tree")}, indent=1))


if __name__ == "__main__":
    main()
