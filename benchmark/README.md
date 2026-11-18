# Independent validation benchmark

This is the validation benchmark of Appendix C.4 of the paper. It is a separate code base that does not
import `robbins_lb` (or anything else in this repository) and shares no code with it. It provides:

- exact or high-precision values of the full-information optimal values v_1, ..., v_4;
- two-sided enclosures of the Bruss–Ferguson truncated values W_n^(m), m ≤ 5, n ≤ 17;
- a reproduction of values of the (d,k)-abstraction of Brice, Bruss, Majumdar and Raskin (arXiv 2608.27419).

Only the values v_1..v_4 (`results/exact.json`) are used elsewhere in this repository, as anchors in
`tests/test_relax.py` and in `replication/`. Nothing in Theorem 6.1 depends on this benchmark.

The code is a copy of the benchmark as frozen before the certified runs. Three docstrings were reworded;
no code was changed. `results/` holds the production outputs and
run logs from November 2026.

## Definitions

- X_1, ..., X_n are i.i.d. U(0,1); smaller is better.
- R_k = Σ_i 1(X_i ≤ X_k) is the absolute rank.
- v_n = inf_τ E R_τ; v_n is nondecreasing and V = lim v_n (Bruss and Ferguson 1993, Theorem 2).
- **W_n^(m)** = inf_τ E Y_τ with Y_k = 1 + min{m, Σ_{i<k} 1(X_i < X_k)} + (n − k) X_k
  (Bruss–Ferguson eq. (4.4)). Only the past part of the rank is truncated.
  W_n^(m) ≤ v_n, with equality when m ≥ n − 1. Their Table 3 tabulates W.
- Recursion (Bruss–Ferguson eq. (4.5)), over the m smallest past values y_1 ≤ ... ≤ y_m:

```
V_{k+1}(y) = Σ_{j=0}^{m-1} ∫_{y_j}^{y_{j+1}} min{ kx + j , V_k(y_1..y_j, x, y_{j+1}..y_{m-1}) } dx + (1 − y_m) V_k(y),
y_0 = 0,  V_0 ≡ m,  W_n^(m) = 1 + V_n(1,…,1).
```

## Methods

| quantity | method | file |
|---|---|---|
| v_2 = 5/4 | closed form | `smalln.py` |
| v_3 | exact (SymPy): 341/144 − 13√13/48 | `smalln.py` |
| v_4 | mpmath, 30–40 digits (exact inner integrals, adaptive outer Gauss–Legendre) | `smalln.py` |
| W_n^(1), n ≤ 17 | exact piecewise-polynomial propagation of (4.5) for m = 1, 60–90 digits | `m1_exact.py` |
| W_n^(m), m ≤ 5 | lower/upper grid recursions with directed rounding | `truncated_dp.py` |
| same, slow reference | literal `mpmath.iv` implementation (tests only) | `ivref.py` |
| rule values | Monte Carlo of explicit stopping rules under the definition (4.4) | `policy_sim.py` |
| (d,k)-abstraction | backward induction over box tuples | `kbest.py` |

The enclosures rest on a discretisation lemma, which is only sketched here and in Appendix C.4 of the
paper:

- The value function of (4.5) is nonincreasing in each y_i.
- Restricting the states to grid tuples and inserting the right (left) end of the arrival's cell gives a
  lower (upper) recursion that brackets the true value.
- The grids are dyadic, so k·t + j, t_{c+1} − t_c and t_c + t_{c+1} are exact. Every other operation is
  rounded outward with `nextafter`, so the lower pass returns a number ≤ W and the upper pass a number
  ≥ W.

Production grids are power grids t_i = (i/G)^1.5:

| m | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| G | 2^18 | 8192 | 768 | 224 | 104 |

The enclosure width is of first order in the grid spacing (≈ 2/G at n = 17).

## Results

Exact values:

| n | v_n |
|---|---|
| 1 | 1 |
| 2 | 5/4 |
| 3 | 341/144 − 13√13/48 = 1.39155208511739178866909979562… |
| 4 | 1.49329759337122544963488318142… (dps 30 and 40 agree to 28 digits) |

Enclosures of W_17^(m) (all n = 1..17 are in `results/m*.json`):

| m | lower | upper | printed in Bruss–Ferguson Table 3 |
|---|---|---|---|
| 1 | 1.438557 | 1.438561 | 1.4385 |
| 2 | 1.608955 | 1.609114 | 1.6092 |
| 3 | 1.703652 | 1.705737 | 1.7070 |
| 4 | 1.762131 | 1.770193 | 1.7668 |
| 5 | 1.799486 | 1.818175 | 1.8090 |

- The corresponding enclosures at n = 3, 4 contain the exact v_3 and the high-precision reference for v_4.
- The exact m = 1 solver gives W_17^(1) = 1.438559399184793411659…
- Three printed entries of Bruss–Ferguson Table 3 lie outside these enclosures: W_17^(1), W_17^(2) and
  W_17^(3). As in the paper, we do not claim that the published values are wrong. The discretisation
  lemma has only been sketched, and the source of the differences (which could, for instance, come from
  discretisation conventions) has not been investigated.

(d,k)-abstraction. The 14 published values of Table 1 of arXiv 2608.27419 that were checked (n = 3, 4, 5, 10)
are reproduced to the printed digits by the authors' public code (`bottom-up.py` of the repository
`anirban11/Robbins_MDP`, read but not run). A literal implementation of Definition 5 / eq. (10) as printed
gives slightly different values at n ≥ 4. The difference lies in the number of forgotten draws in the loss
terms; see `kbest.py`. These values are approximations and are not used as bounds anywhere in this repository.

## Reproduce

From this directory:

```
python -I -X utf8 run.py m1 m2 m3 series     # ~25 min
python -I -X utf8 run.py m4                  # ~20 min, ~2.7 GB RAM
python -I -X utf8 run.py m5                  # ~22 min, ~2.8 GB RAM
python -I -X utf8 run.py exact kbest         # exact ~2 min; kbest ~90 min single-threaded
python -I -X utf8 run.py policysim           # ~2 min (seeds 11, 22, 33, 44, 55)
python -I -X utf8 -m pytest -q -p no:cacheprovider robbins_bench/tests
python -I -X utf8 -c "import sys; sys.path.insert(0, '.'); from robbins_bench.report import main; main()"   # summary tables
```

Each task writes `results/<task>.json` and overwrites the recorded output. All runs are deterministic
except `policysim`, which uses the fixed seeds listed above. The test suite (56 tests, about 45 s) reads
`results/*.json`. It checks the fast recursion against the `mpmath.iv` reference, the independent
solvers against the enclosures, directed rounding against 120-digit arithmetic, and the recorded
production values. Tested with Python 3.12.16, NumPy 2.5.4, numba 0.68.1, mpmath 1.3.0 and SymPy 1.14.0.
