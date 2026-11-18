"""Information-relaxation dual lower bounds and primal upper bounds for Robbins' problem.

Usage:  python -X utf8 run_dual.py [config.json]
Writes checkpoints to ./ckpt/ and the final table to ./results.json (next to this script).
Every number here is a floating-point Monte Carlo estimate, NOT certified and not part of the proof.
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


def comb_dual(C, M2, M3, lam):
    return np.min(C - lam[0] * M2 - lam[1] * M3, axis=1)


def fit_lambda(C, M2, M3):
    """maximise mean_p min_j (c - l2 M2 - l3 M3): concave in (l2, l3)."""
    f = lambda lam: -float(np.mean(comb_dual(C, M2, M3, lam)))
    best = None
    for x0 in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]):
        r = minimize(f, np.array(x0), method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-7, maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    return best.x.tolist(), -best.fun


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(sum=float(a.sum()), sumsq=float((a * a).sum()), cnt=int(a.size))


def merge(acc, st):
    if acc is None:
        return dict(st)
    return dict(sum=acc["sum"] + st["sum"], sumsq=acc["sumsq"] + st["sumsq"], cnt=acc["cnt"] + st["cnt"])


def mean_se(acc):
    m = acc["sum"] / acc["cnt"]
    var = acc["sumsq"] / acc["cnt"] - m * m
    var *= acc["cnt"] / (acc["cnt"] - 1)
    return m, math.sqrt(max(var, 0.0) / acc["cnt"])


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    numba.set_num_threads(cfg["threads"])
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception as e:  # pragma: no cover
        log("could not lower priority:", e)
    ck_dir = os.path.join(HERE, "ckpt")
    os.makedirs(ck_dir, exist_ok=True)
    results = {}
    for n in cfg["n_list"]:
        beta = default_beta(n)
        t0 = time.time()
        g2, W2 = solve_dp2(n, cfg["G2"], beta)
        g3, W3 = solve_dp3(n, cfg["G3"], beta)
        dp2v = float(W2[0][-1, -1]); dp3v = float(W3[0][-1, -1, -1])
        log(f"n={n}: DP2={dp2v:.5f} DP3={dp3v:.5f}  ({time.time()-t0:.1f}s)")
        # ---- lambda fit on an independent training set (never reused for evaluation) ----------
        lam_path = os.path.join(ck_dir, f"lam_n{n}.json")
        if os.path.exists(lam_path):
            lam_info = json.load(open(lam_path))
        else:
            rng = np.random.default_rng(cfg["train_seed"] + n)
            X = rng.random((cfg["train_paths"], n))
            _, _, _, _, C, M2 = run_sim(2, X, W2, g2, beta, record=True)
            _, _, _, _, C3, M3 = run_sim(3, X, W3, g3, beta, record=True)
            assert np.allclose(C, C3)
            lam, val = fit_lambda(C, M2, M3)
            lam_info = dict(lam=lam, train_value=val, train_paths=cfg["train_paths"],
                            train_seed=cfg["train_seed"] + n)
            json.dump(lam_info, open(lam_path, "w"), indent=1)
        lam = np.array(lam_info["lam"])
        log(f"n={n}: lambda={lam.round(4).tolist()} (train value {lam_info['train_value']:.5f})")
        # ---- evaluation chunks (common random numbers for all V-tilde choices) -----------------
        acc = {}
        nchunks = cfg["paths"] // cfg["chunk"]
        for ch in range(nchunks):
            cp = os.path.join(ck_dir, f"n{n}_chunk{ch:03d}.json")
            if os.path.exists(cp):
                st = json.load(open(cp))
            else:
                rng = np.random.default_rng([cfg["seed"], n, ch])
                X = rng.random((cfg["chunk"], n))
                d2, p2, tau2, Mn2, C, M2 = run_sim(2, X, W2, g2, beta, record=True)
                d3, p3, tau3, Mn3, C3, M3 = run_sim(3, X, W3, g3, beta, record=True)
                dc = comb_dual(C, M2, M3, lam)
                st = dict(dual_k2=stats(d2), dual_k3=stats(d3), dual_comb=stats(dc),
                          primal_k2=stats(p2), primal_k3=stats(p3),
                          gap_k2=stats(p2 - d2), gap_k3=stats(p3 - d3), gap_best=stats(p3 - dc),
                          Mn_k2=stats(Mn2), Mn_k3=stats(Mn3),
                          tau_k2=stats(tau2), tau_k3=stats(tau3))
                json.dump(st, open(cp, "w"))
            for key, v in st.items():
                acc[key] = merge(acc.get(key), v)
            if ch % 5 == 4 or ch == nchunks - 1:
                m, s = mean_se(acc["dual_comb"])
                log(f"n={n} chunk {ch+1}/{nchunks}: dual_comb {m:.5f} +- {s:.5f}")
        res = dict(n=n, beta=beta, DP2=dp2v, DP3=dp3v, lam=lam.tolist(), paths=acc["dual_k2"]["cnt"])
        for key, a in acc.items():
            m, s = mean_se(a)
            res[key] = [m, s]
        results[n] = res
        json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
        log(f"n={n} done in {time.time()-t0:.0f}s: " + ", ".join(
            f"{k}={v[0]:.5f}+-{v[1]:.5f}" for k, v in res.items() if isinstance(v, list) and k != "lam"))
    log("ALL DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "config.json"))
