# Archives that could not be retrieved

Retrieval date: 2026-11-05. Eight Nazario files could not be retrieved; the reasons below are those recorded in
`datasets/download_manifest.json` (four attempts per file). Nothing was substituted and no security control was
bypassed.

- Seven files: the institution's web proxy (TLS inspection by an organisation-internal certificate authority)
  answered with HTTP 403 and a block page stating "Threat found: Virus"; some attempts ended with the connection
  closed instead. The downloader rejected every block page.
- phishing3.mbox: no block page was recorded; each of the four attempts failed when the partial download was
  opened for writing on the local disk (operating-system error 22, "Invalid argument"). The cause was not
  determined further.

| Collection | File | URL | Size on index | Index date | Recorded failure |
|---|---|---|---|---|---|
| Nazario earlier mailboxes | phishing0.mbox | https://monkey.org/~jose/phishing/phishing0.mbox | 3.0 MB | 2005-06-13 | HTTP 403 block page |
| Nazario earlier mailboxes | phishing1.mbox | https://monkey.org/~jose/phishing/phishing1.mbox | 4.0 MB | 2005-11-15 | HTTP 403 block page |
| Nazario earlier mailboxes | phishing2.mbox | https://monkey.org/~jose/phishing/phishing2.mbox | 9.9 MB | 2006-08-07 | HTTP 403 block page |
| Nazario earlier mailboxes | phishing3.mbox | https://monkey.org/~jose/phishing/phishing3.mbox | 19 MB | 2007-08-07 | local write error (errno 22) |
| Nazario earlier mailboxes | 20051114.mbox | https://monkey.org/~jose/phishing/20051114.mbox | 3.9 MB | 2014-02-10 | HTTP 403 block page |
| Nazario annual | phishing-2022 | https://monkey.org/~jose/phishing/phishing-2022 | 4.6 MB | 2023-01-31 | HTTP 403 block page |
| Nazario annual | phishing-2023 | https://monkey.org/~jose/phishing/phishing-2023 | 7.4 MB | 2024-01-01 | HTTP 403 block page |
| Nazario annual | phishing-2024 | https://monkey.org/~jose/phishing/phishing-2024 | 15 MB | 2025-01-05 | transfer cut 1,892 bytes short, then HTTP 403 block page |

The analyses therefore use the Nazario annual mailboxes 2015 to 2021 and 2025 (8 of 11 annual files) and none
of the earlier mailboxes. `private-phishing4.mbox` is excluded by design (its name marks it private and the
README does not cover it).

## How to add the missing files later

1. On a computer or network without this filter (or after an IT exception for
   `https://monkey.org/~jose/phishing/`), download the files above with a browser or `curl -O`. They contain
   live phishing content: do not open them in a mail client and do not open attachments.
2. Copy them unchanged to `data/raw/nazario_recent/` (the three annual files) and `data/raw/nazario_classic/`
   (the five earlier files). Endpoint protection on the computer may quarantine them; check the sizes.
3. Run `python -I code/scripts/acquire_data.py --register-local` to format-check, hash and register them.
4. Write a dated protocol amendment that fixes their role before any of their outcomes are seen, then rebuild
   the corpus (`code/scripts/build_corpus.py`) and rerun the grid in a new run directory.
