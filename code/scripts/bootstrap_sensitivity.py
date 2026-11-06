"""Sensitivity of the confirmatory decisions to the bootstrap seed (a check that the analysis plan did not declare).

The eight confirmatory tests are recomputed with the declared procedure (two-level paired bootstrap, 2,000 draws,
Holm adjustment over the eight tests) under the declared seed and eleven further seeds, and the per-replicate
differences in average precision are listed. Each seed runs in its own worker process; its result depends only on
the seed. Writes evidence/metrics/bootstrap_seed_sensitivity.csv and
evidence/metrics/confirmatory_replicate_differences.csv. The reported decisions are those of the declared seed.
On the study laptop one seed takes about 25 minutes of processor time.

Usage: python -I code/scripts/bootstrap_sensitivity.py [--workers 6]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))

from phishstudy.runtime import disable_power_throttling  # noqa: E402

disable_power_throttling()

import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from analyze import CONFIRMATORY, CONFIRMATORY_POPULATIONS, PRIMARY, load_scores  # noqa: E402
from phishstudy.analysis import population  # noqa: E402
from phishstudy.stats import holm, paired_bootstrap  # noqa: E402

DECLARED_SEED = 20261105
SEEDS = [DECLARED_SEED, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def run_seed(job) -> list[dict]:
    """All eight tests and their Holm adjustment for one bootstrap seed."""
    seed, tests, n_boot = job
    res = [dict(population=pop, model_a=m_a, model_b=m_b, **paired_bootstrap(reps, n_boot=n_boot, seed=seed))
           for pop, m_a, m_b, reps in tests]
    for r, h in zip(res, holm([r["p_value"] for r in res])):
        r["holm_p"] = float(h)
        r["reject_at_0.05"] = bool(h < 0.05)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=CODE.parent)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    root = a.project.resolve()
    runs, out = root / "evidence" / "runs", root / "evidence" / "metrics"
    metrics = pd.read_parquet(out / "run_metrics.parquet")
    sel = metrics[metrics.alpha.isna() & metrics.population.eq("in_distribution")]
    chosen = (sel.drop_duplicates(["pairing", "replicate", "variant"])
              .set_index(["pairing", "replicate", "variant"])["selected_weighting"])
    tests, diffs = [], []
    for pop in CONFIRMATORY_POPULATIONS:
        for m_a, m_b in CONFIRMATORY:
            reps = []
            for rep in sorted(metrics[metrics.pairing.eq(PRIMARY)].replicate.unique()):
                za = population(load_scores(runs, PRIMARY, rep, m_a, chosen[(PRIMARY, rep, m_a)]), pop)
                zb = population(load_scores(runs, PRIMARY, rep, m_b, chosen[(PRIMARY, rep, m_b)]), pop)
                m = za.merge(zb, on=["record_id", "label"], suffixes=("_a", "_b"), validate="one_to_one")
                m = m.sort_values("record_id")
                reps.append((m.label.to_numpy(), m.score_a.to_numpy(), m.score_b.to_numpy()))
                diffs.append(dict(population=pop, model_a=m_a, model_b=m_b, replicate=int(rep),
                                  difference=float(average_precision_score(m.label, m.score_a)
                                                   - average_precision_score(m.label, m.score_b))))
            tests.append((pop, m_a, m_b, reps))
    rows = []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(SEEDS))) as pool:
        for seed, res in zip(SEEDS, pool.map(run_seed, [(s, tests, a.n_boot) for s in SEEDS])):
            rows += res
            print(f"seed {seed} done", flush=True)
    pd.DataFrame(rows).to_csv(out / "bootstrap_seed_sensitivity.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "confirmatory_replicate_differences.csv", index=False)
    print("written bootstrap_seed_sensitivity.csv and confirmatory_replicate_differences.csv")


if __name__ == "__main__":
    main()
