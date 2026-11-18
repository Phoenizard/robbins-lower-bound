"""Diagnostic: per-step martingale-increment means D_s = Vt_s - E[Vt_s | F_{s-1}], with SEs.

A bias in any step would show as |z| >> 3.  Usage: python check_martingale.py n k paths seed
"""
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

n, k, paths, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
numba.set_num_threads(8)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
beta = default_beta(n)
g, W = solve_dp2(n, 200, beta) if k == 2 else solve_dp3(n, 80, beta)
S = np.zeros(n); SS = np.zeros(n); N = 0
Smn = 0.0; SSmn = 0.0
chunk = 100000
for ch in range(paths // chunk):
    X = np.random.default_rng([seed, n, ch]).random((chunk, n))
    d, pr, tau, Mn, C, M = run_sim(k, X, W, g, beta, record=True)
    D = np.diff(np.concatenate([np.zeros((chunk, 1)), M], axis=1), axis=1)
    S += D.sum(0); SS += (D * D).sum(0); N += chunk
    Smn += Mn.sum(); SSmn += (Mn * Mn).sum()
m = S / N; se = np.sqrt((SS / N - m * m) / N)
z = m / se
out = dict(n=n, k=k, paths=N, seed=seed, mean=m.tolist(), se=se.tolist(), z=z.tolist(),
           Mn_mean=Smn / N, Mn_se=math.sqrt((SSmn / N - (Smn / N) ** 2) / N),
           sum_step_means=float(m.sum()), max_abs_z=float(np.max(np.abs(z))))
for s in range(n):
    print(f"s={s+1:3d} mean={m[s]: .2e} se={se[s]:.2e} z={z[s]: .2f}")
print("Mn", out["Mn_mean"], "+-", out["Mn_se"], " max|z|", out["max_abs_z"])
json.dump(out, open(os.path.join(HERE, f"martingale_check_n{n}_k{k}_seed{seed}.json"), "w"), indent=1)
