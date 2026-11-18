"""Policy-simulation check (task 'policysim'): python -I -X utf8 run.py policysim"""
import time

import numpy as np

from .grid import make_grid
from .policy_sim import simulate_policy
from .truncated_dp import value_tables

CASES = [  # (m, n, G for the policy tables, paths per seed)
    (2, 17, 512, 4_000_000),
    (3, 17, 128, 4_000_000),
    (4, 17, 64, 4_000_000),
    (5, 17, 40, 4_000_000),
    (3, 9, 128, 4_000_000),
]
SEEDS = [11, 22, 33, 44, 55]


def run():
    out = []
    for m, n, G, npaths in CASES:
        t = make_grid(G, "power", 1.5)
        lo = value_tables(m, t, n - 1, True)
        hi = value_tables(m, t, n - 1, False)
        mid = [0.5 * (a + b) for a, b in zip(lo, hi)]
        t0 = time.time()
        res = [simulate_policy(m, n, t, mid, npaths, s) for s in SEEDS]
        means = np.array([r[0] for r in res])
        ses = np.array([r[1] for r in res])
        pooled_mean = float(means.mean())
        pooled_se = float(np.sqrt((ses ** 2).sum()) / len(ses))
        rec = dict(m=m, n=n, G_policy=G, paths_per_seed=npaths, seeds=SEEDS, means=means.tolist(),
                   ses=ses.tolist(), pooled_mean=pooled_mean, pooled_se=pooled_se,
                   upper_3se=pooled_mean + 3 * pooled_se, seconds=time.time() - t0)
        print(rec, flush=True)
        out.append(rec)
    return out
