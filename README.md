# Phishing email detection across public mail collections

Code and evidence for a source-disjoint evaluation of phishing e-mail detectors on real public mail: two phishing
collections (Nazario annual mailboxes, Phishing Pot), two legitimate collections (SpamAssassin non-spam, Enron) and
a modern legitimate collection (Apache user mailing lists) that never trains, selects or calibrates a detector and
enters only the evaluation and two collection-identity probes. Ten model variants, including a
feature-token Transformer, a one-token attention control, an MLP, tree ensembles, logistic regression and a TF-IDF
text model, are trained on each of four phishing-legitimate pairings and tested on collections they never saw, with
Neyman-Pearson false-positive thresholds calibrated on held-out legitimate mail.

No message text is distributed. The repository holds code, frozen split assignments, hashed unit identifiers with
labels and engineered features, the saved score of every evaluation and calibration unit for every fit, metrics,
figures and audit outputs.

## Results (selected weighting, mean over four training pairings and five replicates)

| Variant | AP, ID | AP, both shift | FPR at 1%, L' (%) | FPR at 1%, Apache lists (%) |
|---|---:|---:|---:|---:|
| Logistic regression | 0.989 | 0.881 | 6.0 | 32.6 |
| Random forest | 0.992 | 0.932 | 10.3 | 38.9 |
| Gradient boosting | 0.992 | 0.885 | 11.7 | 39.4 |
| MLP | 0.992 | 0.815 | 9.2 | 30.5 |
| Single-token attention | 0.990 | 0.875 | 7.2 | 32.7 |
| Feature-token Transformer | 0.990 | 0.857 | 10.4 | 35.8 |
| FTT without link features | 0.988 | 0.864 | 7.6 | 32.9 |
| FTT without urgency count | 0.987 | 0.880 | 10.6 | 36.3 |
| Gradient boosting, format only | 0.985 | 0.847 | 9.8 | 34.4 |
| TF-IDF logistic | 0.996 | 0.819 | 9.7 | 2.7 |

Column definitions: ID is the in-distribution population (test units of the training phishing and legitimate
collections); both shift uses all units of the other phishing and the other legitimate collection; AP is average
precision (non-interpolated area under the precision-recall curve); FPR at 1% is the realised false-positive rate, in
percent, at the threshold calibrated for a 1 percent budget on held-out legitimate units of the training collection,
measured on all units of the other legitimate collection (L') and on all Apache list units.

## Requirements

Tested on Windows 11 with Python 3.12.10 on an Intel Core Ultra 7 155U laptop (14 logical
processors, 15.5 GiB memory), CPU only, with the direct dependencies pinned in `requirements.txt` (the full environment is listed in
`environment-freeze.txt`):

```
python -m venv .venv
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
.venv\Scripts\python -m pip install -r requirements.txt
```

The code is pure Python. On Linux or Google Colab the same commands apply with `python` in place of
`.venv\Scripts\python`; these platforms were not tested.

## Reproduce

Fast path, from the shipped evidence (no mail archives needed; the run time per step is written to
`evidence/logs/reproduce_timing.json`):

```
.venv\Scripts\python -I code/scripts/reproduce.py
```

This restores the saved scores from `evidence/packed`, then recomputes every metric, the eight declared tests, the
independent audit (including the 2,000-draw bootstrap), all figures and the HTML report. Adding `--seed-check`
also repeats the confirmatory bootstrap under eleven further seeds (`code/scripts/bootstrap_sensitivity.py`, about
an hour with six worker processes), a check that the analysis plan did not declare; the reported decisions are
those of the declared seed.

Full path, from the original archives (internet access and several hours on a laptop CPU):

```
.venv\Scripts\python -I code/scripts/reproduce.py --full
```

The full path runs the steps that produced the shipped evidence: it downloads each archive from its original host and
records its SHA-256 digest, rebuilds the corpus and the splits, refits every model with deterministic CPU kernels, runs
the collection-identity probes and then the fast path. These steps rewrite the files in `datasets/`, and archives can
change at their hosts, so keep copies of the shipped `datasets/download_manifest.json` and
`datasets/corpus_summary.json` and compare the archive digests and the unit and split hashes with them before
comparing results. Archives that a network filter blocks must be fetched
manually (`datasets/BLOCKED_ARCHIVES.md`) and registered with `.venv\Scripts\python -I code/scripts/acquire_data.py
--register-local`.

Tests: `.venv\Scripts\python -I -m pytest code/tests -q` (the integration tests run only when the corpus has been
rebuilt locally).

## Layout

- `code/phishstudy/`: `sources.py` (archive registry), `acquire.py` (HTTPS retrieval with digests), `parse.py`
  (MIME parsing without executing content), `features.py` (23 engineered features, text normalisation), `dedup.py`
  (exact, template and MinHash near-duplicate grouping), `corpus.py` (screening and unit freeze), `splits.py`,
  `models.py`, `experiment.py` (training pairings, thresholds, saved scores), `stats.py`, `analysis.py`,
  `probe.py` (collection-identity probes), `verify.py` (independent recomputation), `figures.py`,
  `evidence_pack.py` (compact storage of saved scores), `runtime.py`.
- `code/scripts/`: the command-line entry points used above, plus `run_experiments.py`, `run_probes.py`,
  `build_corpus.py`, `check_phishing_pot.py` (file-by-file check against the Git tree of the pinned commit),
  `bootstrap_sensitivity.py` (confirmatory decisions under further bootstrap seeds), `make_label_review_sheet.py`
  and `write_provenance.py`.
- `code/notebooks/`: the notebook builder and executor; the executed notebook is distributed with the article.
- `code/tests/`: unit and integration tests.
- `datasets/`: provenance, retrieval manifest, frozen unit manifest with engineered features, splits and the
  label review sheet (no text).
- `evidence/`: packed scores of every fit and probe, metrics, figures, logs and the HTML report.
- `research/Preregistered_Protocol.md`: the analysis plan fixed before any fit, with its amendment.
- `worklog/protocol_freeze.json`: SHA-256 digests and UTC times of the plan's freeze and amendment;
  `worklog/deviations.md`: every later change and choice, with its reason and time.
- `audit/Numeric_Evidence_Checks.json`: output of the independent verification.

## Data sources and terms

- Nazario phishing corpus, https://monkey.org/~jose/phishing/ (CC BY 4.0).
- Phishing Pot, https://github.com/rf-peixoto/phishing_pot, commit 78be141545ed9e56eefb857808ac78e83eb81b81
  (CC BY-NC 4.0; the curator asks for credit linking to the repository and to https://www.linkedin.com/in/rf-peixoto).
- Apache SpamAssassin public corpus, https://spamassassin.apache.org/old/publiccorpus/ (message copyright remains
  with the senders; the messages must not be sent into a live mail system).
- CMU Enron corpus, https://www.cs.cmu.edu/~enron/ (7 May 2015 version; respect the privacy of the people involved).
- Apache Software Foundation mailing-list archives, https://lists.apache.org/ (Public Forum Archive Policy).

Derived files are released for non-commercial research use because the Phishing Pot licence is non-commercial.

## Licence

Code: MIT (see `LICENSE`). Mail archives: the terms of their publishers above.
