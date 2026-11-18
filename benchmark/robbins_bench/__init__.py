"""robbins_bench: independent validation benchmark for Robbins' problem (exact v_1..v_4, enclosures of W_n^(m)).

Modules
-------
grid          exact dyadic grids on [0, 1]
truncated_dp  rigorous lower/upper enclosures of Bruss-Ferguson W_n^(m) via their recursion (4.5),
              including the directed-rounding (one-sided interval) primitives for numba
m1_exact      independent high-precision solver for the m = 1 column (piecewise polynomials, mpmath)
smalln        exact / high-precision full-information values v_1..v_4
ivref         slow pure-mpmath.iv reference implementation of truncated_dp (for tests)
kbest         re-implementation of the (d,k)-abstraction M_{n,d,k} of Brice-Bruss-Majumdar-Raskin
"""
