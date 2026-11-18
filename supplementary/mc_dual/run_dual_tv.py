"""Variant of run_dual.py with time-varying martingale weights (needs ckpt/lam_n*.json from run_dual.py).

M_j = sum_{s<=j} (l2_s * D2_s + l3_s * D3_s), where D^k_s are the k-memory martingale increments and
l_s are DETERMINISTIC weights.  Hence M is still a martingale and v_n >= E[min_j (c_j - M_j)].
The weights are fitted on an independent training set by maximising a soft-min surrogate (L-BFGS,
temperature annealed), then frozen and evaluated with the hard min on the SAME evaluation paths
as run_dual.py (seed (20261112, n, chunk)).  Monte Carlo; not certified.
Usage: python -X utf8 run_dual_tv.py config.json
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numba  # noqa: E402
import psutil  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from mcdual.dp2 import solve_dp2  # noqa: E402
from mcdual.dp3 import solve_dp3  # noqa: E402
from mcdual.dual import run_sim  # noqa: E402
from mcdual.grid import default_beta  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def increments(M):
    return np.diff(np.concatenate([np.zeros((M.shape[0], 1)), M], axis=1), axis=1)


def increments_pair(M2, M3):
    """D2, D3 with the last-step duplicate removed: at s = n both equal c_n - E[c_n | F_{n-1}]
    (identical up to rounding), which makes (l2_n, l3_n) a flat direction.  Keep it in D2 only."""
    D2, D3 = increments(M2), increments(M3)
    D3[:, -1] = 0.0
    return D2, D3


def hard_dual(C, D2, D3, l2, l3):
    M = np.cumsum(D2 * l2[None, :] + D3 * l3[None, :], axis=1)
    return np.min(C - M, axis=1)


def fit_tv(C, D2, D3, l0, temps=(0.1, 0.03, 0.01, 0.003)):
    n = C.shape[1]

    def obj(theta, tau):
        l2, l3 = theta[:n], theta[n:]
        Y = C - np.cumsum(D2 * l2[None, :] + D3 * l3[None, :], axis=1)
        ymin = Y.min(axis=1, keepdims=True)
        E = np.exp(-(Y - ymin) / tau)
        Z = E.sum(axis=1, keepdims=True)
        soft = ymin[:, 0] - tau * np.log(Z[:, 0])       # soft-min <= hard min
        w = E / Z                                        # d soft / d Y_j
        R = np.cumsum(w[:, ::-1], axis=1)[:, ::-1]       # sum_{j>=s} w_j
        # dY_j / d l_s = -D_s 1[s<=j]  -> d soft / d l_s = -sum_{j>=s} w_j D_s
        g2 = -(R * D2).mean(axis=0)
        g3 = -(R * D3).mean(axis=0)
        return -soft.mean(), -np.concatenate([g2, g3])

    theta = np.asarray(l0, dtype=float)
    for tau in temps:
        r = minimize(obj, theta, args=(tau,), jac=True, method="L-BFGS-B",
                     bounds=[(-3.0, 3.0)] * (2 * n), options=dict(maxiter=500))
        theta = r.x
    return theta[:n], theta[n:]


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    numba.set_num_threads(cfg["threads"])
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    ck = os.path.join(HERE, "ckpt_tv")
    os.makedirs(ck, exist_ok=True)
    out_path = os.path.join(HERE, "results_tv.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for n in cfg["tv_n_list"]:
        if str(n) in results:
            continue
        t0 = time.time()
        beta = default_beta(n)
        g2, W2 = solve_dp2(n, cfg["G2"], beta)
        g3, W3 = solve_dp3(n, cfg["G3"], beta)
        lam = json.load(open(os.path.join(HERE, "ckpt", f"lam_n{n}.json")))["lam"]
        wpath = os.path.join(ck, f"tv_weights_n{n}.json")
        if os.path.exists(wpath):
            wd = json.load(open(wpath)); l2 = np.array(wd["l2"]); l3 = np.array(wd["l3"])
        else:
            rng = np.random.default_rng(cfg["train_seed"] + 1000 + n)   # fresh training set
            X = rng.random((cfg["tv_train_paths"], n))
            _, _, _, _, C, M2 = run_sim(2, X, W2, g2, beta, record=True)
            _, _, _, _, _, M3 = run_sim(3, X, W3, g3, beta, record=True)
            D2, D3 = increments_pair(M2, M3)
            l0 = np.concatenate([np.full(n, lam[0]), np.full(n, lam[1])])
            l0[n - 1] = lam[0] + lam[1]   # single weight on the (shared) last step
            base = hard_dual(C, D2, D3, l0[:n], l0[n:]).mean()
            assert abs(base - json.load(open(os.path.join(HERE, "ckpt", f"lam_n{n}.json")))["train_value"]) < 0.05
            l2, l3 = fit_tv(C, D2, D3, l0)
            tr = hard_dual(C, D2, D3, l2, l3).mean()
            json.dump(dict(l2=l2.tolist(), l3=l3.tolist(), train_const=float(base), train_tv=float(tr),
                           train_seed=cfg["train_seed"] + 1000 + n, train_paths=cfg["tv_train_paths"]),
                      open(wpath, "w"), indent=1)
            log(f"n={n}: train dual const-lambda {base:.5f} -> time-varying {tr:.5f}")
        S = SS = 0.0; N = 0
        Sg = SSg = 0.0
        for ch in range(cfg["paths"] // cfg["chunk"]):
            X = np.random.default_rng([cfg["seed"], n, ch]).random((cfg["chunk"], n))
            _, p2, _, _, C, M2 = run_sim(2, X, W2, g2, beta, record=True)
            _, p3, _, _, _, M3 = run_sim(3, X, W3, g3, beta, record=True)
            D2, D3 = increments_pair(M2, M3)
            d = hard_dual(C, D2, D3, l2, l3)
            S += d.sum(); SS += (d * d).sum(); N += d.size
            gp = p3 - d
            Sg += gp.sum(); SSg += (gp * gp).sum()
        m = S / N; se = math.sqrt((SS / N - m * m) / N)
        mg = Sg / N; seg = math.sqrt((SSg / N - mg * mg) / N)
        results[str(n)] = dict(dual_tv=[m, se], gap_tv_vs_primal_k3=[mg, seg], paths=N)
        json.dump(results, open(out_path, "w"), indent=1)
        log(f"n={n}: dual_tv = {m:.5f} +- {se:.5f}, gap to k3 primal {mg:.5f} +- {seg:.5f} ({time.time()-t0:.0f}s)")
    log("TV DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "config.json"))
