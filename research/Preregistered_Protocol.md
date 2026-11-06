# Pre-registered analysis protocol

Status: frozen before any model is fitted on the study data. The SHA-256 of this file and the UTC time of
freezing are recorded in `worklog/protocol_freeze.json`. Any later change is listed in
`worklog/deviations.md` with its reason and time; outcomes observed before a change are named there.

This protocol replaces the pre-execution design inherited with the project. The inherited design used one
phishing collection and one legitimate collection, so collection and label were the same variable. The
present design uses two collections per class and evaluates every model on collections it never saw.

## 1. Data

Collections (one label each, assigned by the collector; nothing is relabelled):

| Code | Collection | Label | Archives used |
|---|---|---|---|
| NZ | Nazario phishing mailboxes | phishing | phishing-2015 to phishing-2021 and phishing-2025 (the annual files that the institutional network delivered; phishing-2022 to phishing-2024 and the 2005-2007 mailboxes were blocked by the network's malware filter) |
| PP | Phishing Pot honeypot samples | phishing | repository snapshot at commit 78be141545ed9e56eefb857808ac78e83eb81b81 |
| SA | SpamAssassin public corpus, non-spam | legitimate | 20030228_easy_ham, 20030228_easy_ham_2, 20030228_hard_ham |
| EN | CMU Enron mail (May 7, 2015 version) | legitimate | 10 percent hash sample of messages outside sent, deleted, junk and spam folders |

The Phishing Pot maintainers state that their honeypot addresses also receive spam and fraud; the samples
they publish are their own selection of phishing, so PP labels follow a broader curator definition than the
Nazario inbox. This difference is part of what the cross-collection tests measure and is reported as such.

Inclusion rules, applied identically to every collection: the message decodes; the subject plus visible body
has at least 20 non-space characters; the language identifier (langid.py) labels the text English. The
English restriction is declared because the lexical features (English stop words, an English urgency lexicon)
are English-specific.

Enron sampling: a message is kept when the first 32 bits of SHA-256(member path) modulo 10 equal 0; messages
in folders whose path contains "sent", "deleted", "junk" or "spam" are excluded, because sent mail is outgoing
and deleted or junk folders are the most likely places for unlabelled spam.

Units: exact duplicates (identical canonical text), template duplicates (identical after digits are masked)
and near-duplicates (Jaccard similarity of word 5-shingle sets at least 0.8, candidates from MinHash with 128
permutations in 16 bands, confirmed exactly) are linked across all collections. Groups containing both labels
are quarantined. Groups spanning two or more collections are removed, so that no campaign or message is
shared between collections. One representative per remaining group (smallest record identifier) is the unit of
analysis. Counts of every exclusion are reported.

## 2. Questions

- Q1. How well do the models separate phishing from legitimate mail inside the classic pairing (Nazario
  phishing against SpamAssassin non-spam) and inside the other three pairings?
- Q2. How much of that performance remains when the phishing collection, the legitimate collection, or both
  are replaced by collections never seen during development?
- Q3. Do false-positive thresholds calibrated on held-out legitimate mail of the development collection keep
  their false-positive rate on another legitimate collection?
- Q4. Does attention over feature tokens add value over the same features used by an MLP, by a single-token
  attention control, and by gradient-boosted trees, and how does it compare with a TF-IDF text model?
- Q5. How strongly do the inputs identify the collection within a class, and how much performance is
  reached with formatting features alone?

## 3. Design

Training pairings: NZ-SA (primary; the classic Nazario against SpamAssassin design), NZ-EN, PP-SA, PP-EN.
For a pairing (P, L), P' denotes the other phishing collection and L' the other legitimate collection.

Splits: five replicates with seeds 17, 29, 43, 71 and 101. In each replicate and collection the units are
shuffled with a seed derived from the replicate seed and the collection name and cut 50/10/20/20 into fit,
tune, calibration and test parts. All pairings of one replicate share these splits.

Evaluation populations of a pairing (P, L):
- in-distribution: P test units and L test units;
- phishing shift: all P' units and L test units;
- legitimate shift: P test units and all L' units;
- both shift: all P' units and all L' units.

Fit, tune and calibration units of P and L are never scored as evaluation units. P' and L' are never used for
fitting, selection or thresholds in that pairing.

## 4. Models

Numeric inputs: 23 engineered features (format: characters, words, unique words, non-empty lines, mean word
length, upper-case, digit, punctuation and non-ASCII fractions, HTML part present, HTML only, attachment count;
lexical: English stop words, urgency-lexicon words, e-mail addresses; links: distinct URLs, distinct hosts,
IP-literal URLs, maximum dots in a host, anchors, anchors whose displayed URL host differs from the target host,
anchors with action text, script or form present). Transform log(1 + x), then standardisation fitted on the
fit part.

Text input: subject and visible body, with URLs and e-mail addresses replaced by fixed tokens and the
collectors' own identifiers removed; TF-IDF of word unigrams and bigrams (minimum document frequency 2, at
most 50,000 terms, sublinear term frequency) fitted on the fit part.

| Variant | Family | Inputs | Setting 0 | Setting 1 |
|---|---|---|---|---|
| logistic | logistic regression | 23 features | C = 0.1 | C = 1 |
| random_forest | random forest, 300 trees | 23 features | min leaf 1 | min leaf 5 |
| gradient_boosting | histogram gradient boosting, 300 iterations, rate 0.05 | 23 features | 15 leaves | 31 leaves |
| mlp | two hidden layers | 23 features | width 32 | width 64 |
| single_token_attention | all features as one token, 4-head attention, then MLP head (control) | 23 features | width 32 | width 64 |
| feature_token_transformer | one token per feature plus [CLS], 2 pre-norm blocks, 4 heads | 23 features | width 32 | width 64 |
| ftt_no_url | as above | without the 8 link features | width 32 | width 64 |
| ftt_no_urgency | as above | without the urgency-lexicon count | width 32 | width 64 |
| gb_format_only | gradient boosting (artifact control) | 12 format features | 15 leaves | 31 leaves |
| tfidf_logistic | logistic regression on TF-IDF | text | C = 1 | C = 10 |

Neural models: AdamW (learning rate 0.001, weight decay 0.0001), batch 256, at most 60 epochs, early stopping
with patience 8 on tune-part average precision, gradient clipping at norm 5, deterministic CPU kernels.
Class weighting is a factor with two levels: none, and balanced sample weights. For each variant and weighting,
the setting with the larger tune-part average precision is kept (setting 0 on ties); both weightings are
saved, and the weighting with the larger tune average precision is the selected model of that variant. The
budget is identical for all variants: two settings times two weightings. No synthetic oversampling is used.

## 5. Thresholds

For alpha in {0.001, 0.005, 0.01} and delta = 0.05, the Neyman-Pearson order-statistic threshold of Tong, Feng
and Li (2018) is computed from the calibration-part scores of L only. A budget is unavailable when fewer than
ceil(log(delta)/log(1 - alpha)) benign calibration units exist (2995, 598 and 299 units). The bound applies
to new legitimate messages exchangeable with the calibration messages; under collection shift the realized
false-positive rate is measured, not assumed.

## 6. Endpoints

Primary: average precision (non-interpolated) on each evaluation population; recall at the alpha = 0.01
threshold together with the realized false-positive rate.

Secondary: ROC-AUC, recall and false-positive rate at alpha = 0.005 and 0.001 where available, precision, F1,
Matthews correlation, balanced accuracy, specificity, confusion counts, Brier score, Clopper-Pearson intervals.
Precision is reported with the population prevalence; projected precision at assumed prevalences of 1 percent
and 0.1 percent is an algebraic calculation from measured recall and false-positive rate. For populations
with at least 10,000 legitimate units, 100 subsets with 1 percent phishing prevalence are formed by drawing real
phishing units without replacement and keeping every legitimate unit.

## 7. Confirmatory comparisons

Reference baseline: the MLP on the same 23 features (the closest model without attention), as fixed in the
inherited configuration. Eight two-sided tests on the primary pairing NZ-SA, selected models, populations in-distribution and both
shift, difference in average precision of the feature-token Transformer minus: the MLP; the single-token
attention control; gradient boosting; the TF-IDF logistic model. Inference: two-level paired bootstrap (2000
draws, seed 20261105) resampling replicates and then, within each drawn replicate, positive and negative test
units with replacement; percentile 95 percent intervals; two-sided bootstrap p-values; Holm adjustment over the
eight tests at familywise level 0.05. Everything else is descriptive and reported with replicate means,
standard deviations, ranges and binomial intervals.

## 8. Descriptive analyses

D1 transfer table of all variants over the four pairings and four populations. D2 realized false-positive
rate under legitimate-collection shift at each calibrated threshold, with the share of replicates exceeding
alpha. D3 collection-identity probes: within each class, gradient boosting (all features), gradient boosting
(format features) and TF-IDF logistic regression trained to separate the two collections on fit parts,
selected on tune parts, evaluated on test parts of the same five replicates. D4 format-only control. D5
ablations. D6 weighting comparison. D7 permutation importance (5 shuffles per feature) on the primary pairing,
in-distribution and both-shift populations, for logistic regression, random forest, gradient boosting, MLP and
the feature-token Transformer. D8 mean [CLS] attention per feature token compared with permutation importance
(Spearman correlation), to test whether attention weights track importance. D9 parameters, fitting time,
scoring time and memory. D10 aggregate error analysis without message text.

## 9. Interpretation rules

A model is said to be better on an endpoint only when the declared test rejects after Holm adjustment, or,
for descriptive analyses, when the replicate ranges do not overlap; otherwise the difference is reported as not
established. Results describe these public collections; they are not estimates for a deployed mail service.
No claim of a guarantee is made from an empirical false-positive rate.

## 10. Contingencies

If an archive becomes available later (the blocked Nazario files), it is added only through a dated amendment
that fixes its role before any of its outcomes are seen. If a pairing cannot be formed, its analyses are
reported as unavailable, not replaced.

## Amendment 1 (dated 2026-11-05, written before any model was fitted on study data)

Reason: the dataset review found no public English corpus whose phishing and legitimate messages come from
one mail stream. Both legitimate collections of the frozen design (SpamAssassin 2002, Enron 1999-2002) are
13 to 26 years older than both phishing collections (2015-2026), so every pairing also separates eras. To
measure this, a modern legitimate collection is added for evaluation only.

- Collection AP (legitimate): Apache Software Foundation public archives of six user mailing lists
  (users@tomcat, users@httpd, user@spark, users@kafka, user@cassandra, users@maven), the March archive of each
  year 2015 to 2025, retrieved through the lists.apache.org mbox interface. Messages whose From address has
  an automated local part (jira, git, gitbox, github, buildbot, jenkins, hudson, build, builds, bot, ci,
  review, reviews, or containing noreply, no-reply or notification) are excluded. All other inclusion,
  grouping and unit rules of Section 1 apply unchanged; AP takes part in the cross-collection grouping.
- AP is never used for fitting, selection or thresholds. Two populations are added to every pairing:
  modern legitimate (P test units and all AP units) and phishing shift with modern legitimate (all P' units
  and all AP units). Their analyses are descriptive; the eight confirmatory tests are unchanged.
- Two collection-identity probes are added: SpamAssassin against AP (era within the mailing-list genre) and
  Enron against AP (corporate mail against list mail), with the same models and splits as Section 8 D3.
