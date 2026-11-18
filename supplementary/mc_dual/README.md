# Supplementary: Monte-Carlo information-relaxation dual bounds

**This directory is not part of the proof.** Every number produced here is a floating-point Monte-Carlo
estimate with a standard error. None is certified. The code is independent of `robbins_lb` and is
included only because the column "MC dual" (and "MC rule") of Table 1 of the paper comes from it.

## What it computes

- The cost of stopping at j is c_j = E[R_j | F_j] = r_j + (n − j) X_j, with r_j the relative rank.
- **Dual lower estimate.** For every F-martingale M with M_0 = 0, v_n ≥ E[min_j (c_j − M_j)] (pathwise
  duality: Rogers; Haugh–Kogan; Brown–Smith–Sun).
- The penalty is M_s = Σ_{u≤s} (Ṽ_u − E[Ṽ_u | F_{u−1}]), with Ṽ_u = min(c_u, Ŵ_u(state_u)). Here Ŵ is
  the continuation value of an approximate best-k-memory DP on a grid (k = 2: `mcdual/dp2.py`; k = 3:
  `mcdual/dp3.py`).
- The conditional expectations are computed in closed form, cell by cell (`mcdual/dual.py`). So M is a
  martingale up to floating-point rounding, and the bound is valid for any Ŵ.
- Penalties: k2, k3, and a combination λ2·M^(k2) + λ3·M^(k3). λ is fitted on independent training paths
  (seed 777000 + n) and then frozen. The variant `run_dual_tv.py` uses time-varying deterministic weights.
- **Primal upper estimate.** The rule "stop at the first s with c_s ≤ Ŵ_s(state_s)" is simulated. Its
  mean cost estimates an upper bound on v_n.
- Monte Carlo: 10^6 evaluation paths per n, in chunks of 10^5 with seeds (20261112, n, chunk). Common
  random numbers are used across penalties.

## Recorded results (`recorded/`)

Mean ± 1 standard error, 10^6 paths. Estimates, not bounds with a stated confidence level.

| n | dual, comb | dual, time-varying | primal rule, k3 |
|---|---|---|---|
| 10 | 1.7859 ± 0.0001 | 1.7865 ± 0.0008 | 1.7920 ± 0.0011 |
| 20 | 1.9502 ± 0.0002 | 1.9506 ± 0.0008 | 1.9725 ± 0.0015 |
| 30 | 2.0185 ± 0.0003 | 2.0215 ± 0.0007 | 2.0522 ± 0.0017 |
| 50 | 2.0797 ± 0.0004 | 2.0873 ± 0.0007 | 2.1319 ± 0.0020 |
| 100 | 2.1268 ± 0.0005 | 2.1406 ± 0.0009 | 2.2087 ± 0.0023 |

Files in `recorded/`:

- `results.json` and `results_tv.json`: the full tables, including k2/k3 separately and the DP values.
- `bounds_vs_n.png`: the corresponding figure.
- `replicate_*.json`: replicates on a fresh seed (99) with frozen λ. These agree within about 1–2 SE.
- `martingale_check_*.json`: per-step means of the martingale increments.

Checks done: `test_mcdual.py` (12 tests). The tests compare against v_1 = 1 and v_2 = 5/4, and against
v_3 and v_4 with complete memory. They also compare the closed-form conditional expectations with
adaptive quadrature (error < 1e-9 for k2, about 1e-8 for k3, which is quadrature round-off). Grid
sensitivity at n = 50 was below 1e-4.

## Reproduce

From this directory (8 numba threads, below-normal priority; the original runs took about 12 min for
`run_dual.py` and 24 min for `run_dual_tv.py` on the laptop described in the main README):

```
python -X utf8 run_dual.py config.json        # writes ckpt/ and results.json
python -X utf8 run_dual_tv.py config.json     # needs ckpt/lam_n*.json; writes ckpt_tv/ and results_tv.json
python -X utf8 replicate_fresh_seed.py 50 500000 99
python -X utf8 check_martingale.py 30 2 2000000 4242
python -X utf8 plot_bounds.py
python -m pytest -q test_mcdual.py
```

New outputs are written next to the scripts, so they do not overwrite `recorded/`. `run_dual.py` resumes
from `ckpt/` if it exists.
