"""Write datasets/DATASET_PROVENANCE.csv from the retrieval manifest, the source registry and the corpus summary."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from phishstudy.sources import COLLECTIONS, archive_specs  # noqa: E402

LICENSE_URL = {
    "nazario_classic": "https://monkey.org/~jose/phishing/LICENSE.txt",
    "nazario_recent": "https://monkey.org/~jose/phishing/LICENSE.txt",
    "phishing_pot": "https://github.com/rf-peixoto/phishing_pot/blob/78be141545ed9e56eefb857808ac78e83eb81b81/LICENSE",
    "spamassassin_ham": "https://spamassassin.apache.org/old/publiccorpus/readme.html",
    "enron": "https://www.cs.cmu.edu/~enron/",
    "apache_lists": "https://www.apache.org/foundation/public-archives.html",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    root = p.parse_args().project.resolve()
    manifest = json.loads((root / "datasets" / "download_manifest.json").read_text(encoding="utf-8"))
    files = {r["relative_path"]: r for r in manifest["files"]}
    summary_path = root / "datasets" / "corpus_summary.json"
    archive_log = {}
    log_path = root / "datasets" / "archive_log.json"
    if log_path.exists():
        for a in json.loads(log_path.read_text(encoding="utf-8")):
            archive_log[f"{a['collection']}/{a['archive']}"] = a
    years = json.loads(summary_path.read_text(encoding="utf-8"))["units_years"] if summary_path.exists() else {}
    basis = {"nazario_classic": "published as phishing by the collector (hand classified)",
             "nazario_recent": "published as phishing by the collector (hand classified)",
             "phishing_pot": "published as phishing by the repository maintainer",
             "spamassassin_ham": "classified as non-spam by the corpus collector",
             "enron": "collection membership (legitimate corporate mail)",
             "apache_lists": "collection membership (public mailing lists)"}
    rows = []
    for spec in archive_specs():
        rel = f"{spec['collection']}/{spec['name']}"
        r = files.get(rel, {})
        meta = COLLECTIONS[spec["collection"]]
        a = archive_log.get(rel, {})
        y = years.get(spec["collection"])
        rows.append(dict(
            collection=spec["collection"], archive=spec["name"], url=spec["url"], label=meta["label"],
            label_basis=basis[spec["collection"]],
            collector=meta["collector"], period=meta["period"],
            unit_header_years=(f"median {y['50%']:.0f}, quartiles {y['25%']:.0f} to {y['75%']:.0f} (all units of the "
                               f"collection)" if y else "no units"),
            license=meta["license"],
            license_or_terms_url=LICENSE_URL[spec["collection"]], citation_key=meta["citation_key"],
            retrieval_status=r.get("status", "not attempted"), retrieved_utc=r.get("retrieved_at_utc") or r.get("registered_utc"),
            bytes=r.get("bytes"), sha256=r.get("sha256"),
            failure_reason=("; ".join(dict.fromkeys(e.split(": ", 1)[-1][:160] for e in r.get("errors", [])))
                            if r.get("status") == "failed" else ""),
            candidate_messages=a.get("candidates"), eligible_messages=a.get("eligible"),
            privacy_note="raw messages kept private; only hashed identifiers and derived numbers released",
            real_messages_only="yes (collector-published real mail; no generated records)"))
    out = root / "datasets" / "DATASET_PROVENANCE.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("written", out, len(rows), "rows;", "summary present" if summary_path.exists() else "no summary yet")


if __name__ == "__main__":
    main()
