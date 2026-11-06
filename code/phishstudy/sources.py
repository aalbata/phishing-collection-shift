"""Registry of the original public email archives used in the study.

Each archive belongs to exactly one collection, and every message of a collection carries the label of the
collection: 1 for the phishing collections, whose collectors published the messages as phishing; 0 for the
legitimate collections (SpamAssassin non-spam as classified by its collector; Enron and the Apache lists by
collection membership). No message-level label is assigned by this code.
"""
from __future__ import annotations

NAZARIO = "https://monkey.org/~jose/phishing/"
SPAMASSASSIN = "https://spamassassin.apache.org/old/publiccorpus/"
ENRON = "https://www.cs.cmu.edu/~enron/"
APACHE_API = "https://lists.apache.org/api/mbox.lua"
# Human discussion lists of six Apache projects; one month (March) per year, 2015-2025.
APACHE_LISTS = ["users@tomcat.apache.org", "users@httpd.apache.org", "user@spark.apache.org",
                "users@kafka.apache.org", "user@cassandra.apache.org", "users@maven.apache.org"]
APACHE_YEARS = range(2015, 2026)
PHISHING_POT_REPO = "rf-peixoto/phishing_pot"
# Snapshot of the continuously updated Phishing Pot repository used in this study.
PHISHING_POT_COMMIT = "78be141545ed9e56eefb857808ac78e83eb81b81"

COLLECTIONS = {
    "nazario_classic": dict(
        label=1, period="files last modified 2005-2007 (directory index)", collector="Jose Nazario",
        description="Phishing messages from the collector's personal inbox, hand classified (README); earliest mailboxes",
        license="CC BY 4.0", citation_key="nazario_corpus"),
    "nazario_recent": dict(
        label=1, period="annual files 2015-2025", collector="Jose Nazario",
        description="Phishing messages from the collector's personal inbox, hand classified (README); annual mailboxes",
        license="CC BY 4.0", citation_key="nazario_corpus"),
    "phishing_pot": dict(
        label=1, period="samples up to the pinned commit of 5 November 2026", collector="Phishing Pot maintainer",
        description="Samples received by honeypot addresses, some exposed in data leaks for years; contributors are asked to replace identifying address information with phishing@pot",
        license="CC BY-NC 4.0", citation_key="phishing_pot"),
    "spamassassin_ham": dict(
        label=0, period="archives assembled 28 February 2003", collector="Apache SpamAssassin project",
        description="Non-spam messages (easy_ham, easy_ham_2, hard_ham) of the public mail corpus",
        license="Public research corpus; message copyright remains with original senders",
        citation_key="spamassassin_corpus"),
    "apache_lists": dict(
        label=0, period="March of each year 2015-2025", collector="Apache Software Foundation public mailing-list archives",
        description="User mailing lists of six Apache projects, March of each year; evaluation only",
        license="Public archives under the ASF public mailing list policy; message copyright remains with senders",
        citation_key="apache_lists"),
    "enron": dict(
        label=0, period="May 7, 2015 version of the corpus", collector="FERC release prepared by the CALO project, distributed by CMU",
        description="Corporate mailboxes of about 150 Enron employees (May 7, 2015 version)",
        license="Research distribution; no formal license; privacy-sensitive",
        citation_key="klimt2004enron"),
}


def archive_specs() -> list[dict]:
    """Original files to download, with the collection each one belongs to."""
    specs = []
    for name in ["phishing0.mbox", "phishing1.mbox", "phishing2.mbox", "phishing3.mbox", "20051114.mbox"]:
        specs.append(dict(collection="nazario_classic", name=name, url=NAZARIO + name,
                          kind="mbox", max_bytes=64 * 2**20))
    for year in range(2015, 2026):
        name = f"phishing-{year}"
        specs.append(dict(collection="nazario_recent", name=name, url=NAZARIO + name,
                          kind="mbox", max_bytes=64 * 2**20))
    for name in ["20030228_easy_ham.tar.bz2", "20030228_easy_ham_2.tar.bz2", "20030228_hard_ham.tar.bz2"]:
        specs.append(dict(collection="spamassassin_ham", name=name, url=SPAMASSASSIN + name,
                          kind="tar", max_bytes=16 * 2**20))
    specs.append(dict(collection="phishing_pot", name=f"phishing_pot_{PHISHING_POT_COMMIT[:12]}.zip",
                      url=f"https://codeload.github.com/{PHISHING_POT_REPO}/zip/{PHISHING_POT_COMMIT}",
                      kind="zip", max_bytes=3 * 2**30))
    for lst in APACHE_LISTS:
        for year in APACHE_YEARS:
            local, project = lst.split("@")[0], lst.split("@")[1].split(".")[0]
            specs.append(dict(collection="apache_lists", name=f"{project}_{local}_{year}-03.mbox",
                              url=f"{APACHE_API}?list={lst}&date={year}-03", kind="mbox",
                              max_bytes=200 * 2**20, allow_empty=True, max_seconds=180))
    specs.append(dict(collection="enron", name="enron_mail_20150507.tar.gz",
                      url=ENRON + "enron_mail_20150507.tar.gz", kind="tar", max_bytes=600 * 2**20))
    return specs


def documentation_specs() -> list[dict]:
    raw = f"https://raw.githubusercontent.com/{PHISHING_POT_REPO}/{PHISHING_POT_COMMIT}/"
    docs = [
        ("nazario_classic", NAZARIO + "README.txt", "nazario_README.txt"),
        ("nazario_classic", NAZARIO + "LICENSE.txt", "nazario_LICENSE.txt"),
        ("nazario_classic", NAZARIO, "nazario_index.html"),
        ("spamassassin_ham", SPAMASSASSIN + "readme.html", "spamassassin_readme.html"),
        ("spamassassin_ham", SPAMASSASSIN, "spamassassin_index.html"),
        ("phishing_pot", raw + "README.md", "phishing_pot_README.md"),
        ("phishing_pot", raw + "LICENSE", "phishing_pot_LICENSE.txt"),
        ("phishing_pot", raw + "DATASET_CARD.md", "phishing_pot_DATASET_CARD.md"),
        ("phishing_pot", raw + "CITATION.cff", "phishing_pot_CITATION.cff"),
        ("enron", ENRON, "enron_index.html"),
        ("apache_lists", "https://www.apache.org/foundation/public-archives.html", "apache_public_archives.html"),
    ]
    return [dict(collection=c, url=u, name=n, kind="documentation", max_bytes=4 * 2**20) for c, u, n in docs]
