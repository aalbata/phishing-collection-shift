# Deviations from the pre-registered protocol

1. 2026-11-05, before any model fit: Amendment 1 adds an evaluation-only modern legitimate collection
   (Apache user lists, 2015-2025), two populations and two probes. Reason: the legitimate collections of the
   frozen design are 13 to 26 years older than the phishing collections, so collection and era are
   confounded. No outcome had been observed. See research/Preregistered_Protocol.md, Amendment 1.

2. 2026-11-06, reporting choice made after interim results were seen. The protocol (Section 6) names average
   precision on each evaluation population as a primary endpoint. A trial run of the analysis code on 240 of the
   400 fits showed that the share of phishing differs widely between populations (about 1 to 78 percent), and AP
   depends on that share. Changes between populations are therefore described with ROC-AUC, a declared secondary
   endpoint, while AP is compared across variants within a population and reported for every population
   (Figure 2 and Section S2 of the supplementary material). No analysis, endpoint or test was added or removed; the
   confirmatory tests are unchanged. The manuscript discloses this choice in the endpoints paragraph of the methods.

## Implementation corrections made before the final analysis (the protocol itself is unchanged)

3. 2026-11-05: the subsets with 1 percent phishing were coded for two populations only, whereas the protocol
   declares them for every population with at least 10,000 legitimate units (four populations in this corpus).
   The code was corrected to follow the protocol before this analysis was run on the full grid.

4. 2026-11-06: the fitting process caches the TF-IDF vocabulary and the transformed messages of a replicate, so the
   recorded times of the second weighting run of the text model exclude that work. The compute table takes the
   text model's times from its first weighting run, which includes it; the recorded values are kept alongside.

## Choices that the protocol leaves open, as implemented

5. Weighting ties: when both class weightings reach the same tune-part average precision, the unweighted model is
   the selected one.
6. Language identification reads the first 3,000 characters of the normalised text.
7. MinHash candidates: a band bucket of more than 50 messages contributes only the pairs that include its first
   message.
8. Gradient boosting uses an L2 penalty of 1.0 and no early stopping, for both the full and the format-only variant.

## Checks added after the results were known

9. 2026-11-06, not declared in the protocol. The eight
   confirmatory tests were repeated with eleven further bootstrap seeds (1 to 11) besides the declared seed
   20261105, with the declared procedure and 2,000 draws (code/scripts/bootstrap_sensitivity.py,
   evidence/metrics/bootstrap_seed_sensitivity.csv). Reason: an independent review found two Holm-adjusted
   p-values close to 0.05. The reported decisions remain those of the declared seed; the manuscript reports the
   range of Holm-adjusted p-values of the decisions that changed with the seed.
