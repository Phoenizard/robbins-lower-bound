"""Replicate the main run_dual.py numbers on a fresh seed (frozen lambda from ckpt/lam_n*.json, written by
run_dual.py).  Usage: python -X utf8 replicate_fresh_seed.py n paths seed   (Monte Carlo; not certified)"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numba  # noqa: E402
import psutil  # noqa: E402

from mcdual.dp2 import solve_dp2  # noqa: E402
from mcdual.dp3 import solve_dp3  # noqa: E402
from mcdual.dual import run_sim  # noqa: E402
from mcdual.grid import default_beta  # noqa: E402

n, paths, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
G2, G3 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (200, 80)
numba.set_num_threads(8)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
beta = default_beta(n)
g2, W2 = solve_dp2(n, G2, beta)
g3, W3 = solve_dp3(n, G3, beta)
lam = json.load(open(os.path.join(HERE, "ckpt", f"lam_n{n}.json")))["lam"]
acc = {k: [0.0, 0.0] for k in ("dual_k2", "dual_k3", "dual_comb", "primal_k2", "primal_k3")}
N = 0
chunk = 100000
for ch in range(paths // chunk):
    X = np.random.default_rng([seed, n, ch]).random((chunk, n))
    d2, p2, _, _, C, M2 = run_sim(2, X, W2, g2, beta, record=True)
    d3, p3, _, _, _, M3 = run_sim(3, X, W3, g3, beta, record=True)
    dc = np.min(C - lam[0] * M2 - lam[1] * M3, axis=1)
    for k, a in zip(acc, (d2, d3, dc, p2, p3)):
        acc[k][0] += a.sum(); acc[k][1] += (a * a).sum()
    N += chunk
out = dict(n=n, paths=N, seed=seed, G2=G2, G3=G3, lam=lam)
for k, (s, ss) in acc.items():
    m = s / N
    out[k] = [m, math.sqrt((ss / N - m * m) / N)]
    print(f"{k}: {m:.5f} +- {out[k][1]:.5f}")
json.dump(out, open(os.path.join(HERE, f"replicate_n{n}_seed{seed}_G{G2}-{G3}.json"), "w"), indent=1)
