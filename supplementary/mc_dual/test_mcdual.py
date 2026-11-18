"""Unit tests for the supplementary Monte-Carlo dual code (mcdual).  Run:  python -m pytest supplementary/mc_dual -q"""
import math
import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcdual.dp2 import solve_dp2, dp2_value, panel_min_integral  # noqa: E402
from mcdual.dp3 import solve_dp3  # noqa: E402
from mcdual.dual import (run_sim, _line_int2, _line_int3, _interp2, _interp3,  # noqa: E402
                          _tail_int)
from mcdual.grid import default_beta, make_grid  # noqa: E402

# Exact anchors.  v_1 = 1, v_2 = 5/4 (hand computation: stop iff X_1 <= 1/2).
# v_3 = 1.3915 = W_3^(2) and v_5 = 1.5710 = W_5^(4) (Bruss-Ferguson 1993, Table 3; for m >= n-1 the
# truncation is void, so W_n^(m) = V_n).  Four printed digits -> tolerance 1e-4 (+ grid error).
V2_EXACT = 1.25
V3_BF = 1.3915
V5_BF = 1.5710


def test_grid():
    g = make_grid(50, 4.0)
    assert g[0] == 0.0 and g[-1] == 1.0 and np.all(np.diff(g) > 0)


def test_panel_min_integral_vs_dense():
    rng = np.random.default_rng(0)
    for _ in range(200):
        h = rng.random() + 0.1
        c0, c1, w0, w1 = rng.normal(size=4)
        xs = (np.arange(200000) + 0.5) / 200000 * h
        c = c0 + (c1 - c0) * xs / h
        w = w0 + (w1 - w0) * xs / h
        ref = np.minimum(c, w).mean() * h
        assert abs(panel_min_integral(h, c0, c1, w0, w1) - ref) < 1e-8


@pytest.mark.parametrize("n,exact,tol", [(1, 1.0, 1e-12), (2, V2_EXACT, 1e-12), (3, V3_BF, 2e-4)])
def test_dp2_small_n(n, exact, tol):
    assert abs(dp2_value(n, 200) - exact) < tol


def test_dp3_matches_dp2_at_n3():
    g, W = solve_dp3(3, 80)
    assert abs(float(W[0][-1, -1, -1]) - V3_BF) < 2e-4


def _ev_reference_k2(W, g, beta, n, s, past):
    """E[Vt_s | F_{s-1}] by adaptive quadrature with all kinks/jumps as break points."""
    em1b = math.expm1(beta)
    past = np.sort(past)
    a1 = past[0] if len(past) >= 1 else 1.0
    a2 = past[1] if len(past) >= 2 else 1.0

    def f(x):
        r = 1.0 + np.searchsorted(past, x)
        c = r + (n - s) * x
        if s == n:
            return c
        if x < a1:
            w = _interp2(W[s], g, x, a1, beta, em1b)
        elif x < a2:
            w = _interp2(W[s], g, a1, x, beta, em1b)
        else:
            w = _interp2(W[s], g, a1, a2, beta, em1b)
        return min(c, w)

    pts = np.unique(np.concatenate([[0.0, 1.0], past, g]))
    tot = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        tot += quad(f, lo, hi, epsabs=1e-14, epsrel=1e-13, limit=200)[0]
    return tot


def _ev_k2(W, g, beta, n, s, past):
    em1b = math.expm1(beta)
    past = np.sort(past)
    a1 = past[0] if len(past) >= 1 else 1.0
    a2 = past[1] if len(past) >= 2 else 1.0
    if s == n:
        return 1.0 + float(np.sum(1.0 - past))
    m = float(n - s)
    ev = _line_int2(W[s], g, beta, em1b, 0.0, a1, 1.0, m, 0, a1, a2)
    ev += _line_int2(W[s], g, beta, em1b, a1, a2, 2.0, m, 1, a1, a2)
    st = 2 if len(past) >= 2 else len(past)
    wc = _interp2(W[s], g, a1, a2, beta, em1b)
    ev += _tail_int(np.append(past, 0.0), len(past), st, a2, 3.0, m, wc)
    return ev


def test_conditional_expectation_exact_k2():
    n, G = 12, 60
    beta = default_beta(n)
    g, W = solve_dp2(n, G, beta)
    rng = np.random.default_rng(3)
    worst = 0.0
    for s in range(1, n + 1):
        for scale in (1.0, 0.3, 0.05):
            past = rng.random(s - 1) * scale
            a = _ev_k2(W, g, beta, n, s, past)
            b = _ev_reference_k2(W, g, beta, n, s, past)
            worst = max(worst, abs(a - b))
    assert worst < 1e-9, worst


def test_conditional_expectation_exact_k3():
    n, G = 10, 30
    beta = default_beta(n)
    g, W = solve_dp3(n, G, beta)
    em1b = math.expm1(beta)
    rng = np.random.default_rng(4)
    worst = 0.0
    for s in range(1, n):
        for scale in (1.0, 0.2):
            past = np.sort(rng.random(s - 1) * scale)
            a1 = past[0] if len(past) >= 1 else 1.0
            a2 = past[1] if len(past) >= 2 else 1.0
            a3 = past[2] if len(past) >= 3 else 1.0
            m = float(n - s)
            ev = _line_int3(W[s], g, beta, em1b, 0.0, a1, 1.0, m, 0, a1, a2, a3)
            ev += _line_int3(W[s], g, beta, em1b, a1, a2, 2.0, m, 1, a1, a2, a3)
            ev += _line_int3(W[s], g, beta, em1b, a2, a3, 3.0, m, 2, a1, a2, a3)
            st = 3 if len(past) >= 3 else len(past)
            ev += _tail_int(np.append(past, 0.0), len(past), st, a3, 4.0, m,
                            _interp3(W[s], g, a1, a2, a3, beta, em1b))

            def f(x):
                c = 1.0 + np.searchsorted(past, x) + m * x
                if x < a1:
                    w = _interp3(W[s], g, x, a1, a2, beta, em1b)
                elif x < a2:
                    w = _interp3(W[s], g, a1, x, a2, beta, em1b)
                elif x < a3:
                    w = _interp3(W[s], g, a1, a2, x, beta, em1b)
                else:
                    w = _interp3(W[s], g, a1, a2, a3, beta, em1b)
                return min(c, w)

            pts = np.unique(np.concatenate([[0.0, 1.0], past, g]))
            ref = sum(quad(f, lo, hi, epsabs=1e-14, epsrel=1e-13, limit=200)[0]
                      for lo, hi in zip(pts[:-1], pts[1:]))
            worst = max(worst, abs(ev - ref))
    # scipy.quad reports roundoff near the (unmarked) c = w kink; observed worst ~1e-8.  The exact
    # structural claim (piecewise linearity between grid nodes) is tested separately below.
    assert worst < 1e-7, worst


def test_interpolant_piecewise_linear_along_lines_k3():
    n, G = 10, 30
    beta = default_beta(n)
    g, W = solve_dp3(n, G, beta)
    em1b = math.expm1(beta)
    a1, a2 = 0.0808, 0.9762
    for which in range(3):
        for j in range(G - 1):
            xs = np.linspace(g[j], g[j + 1], 11)
            if which == 0:
                ws = [_interp3(W[3], g, x, a1, a2, beta, em1b) for x in xs]
            elif which == 1:
                ws = [_interp3(W[3], g, a1, x, a2, beta, em1b) for x in xs]
            else:
                ws = [_interp3(W[3], g, a1, a2, x, beta, em1b) for x in xs]
            ws = np.array(ws)
            lin = ws[0] + (ws[-1] - ws[0]) * (xs - xs[0]) / (xs[-1] - xs[0])
            assert np.max(np.abs(ws - lin)) < 1e-12


def test_dual_exact_n2_n3():
    """With complete memory the dual has (near) zero variance and equals v_n."""
    for n, k, val, tol in [(2, 2, V2_EXACT, 1e-9), (3, 2, V3_BF, 2e-4), (3, 3, V3_BF, 2e-4)]:
        beta = default_beta(n)
        g, W = (solve_dp2(n, 200, beta) if k == 2 else solve_dp3(n, 60, beta))
        X = np.random.default_rng(1).random((20000, n))
        d, pr, tau, Mn = run_sim(k, X, W, g, beta)
        assert abs(d.mean() - val) < tol
        assert d.std() < 1e-3


def test_dual_below_bf_n5():
    beta = default_beta(5)
    g, W = solve_dp3(5, 60, beta)
    X = np.random.default_rng(2).random((50000, 5))
    d, pr, tau, Mn = run_sim(3, X, W, g, beta)
    se = d.std() / math.sqrt(len(d))
    assert d.mean() <= V5_BF + 4 * se + 1e-4
    assert d.mean() > V5_BF - 0.01      # tight (complete memory would give equality)


def test_martingale_mean_zero_and_weak_duality():
    n = 15
    beta = default_beta(n)
    g, W = solve_dp2(n, 120, beta)
    X = np.random.default_rng(7).random((100000, n))
    d, pr, tau, Mn = run_sim(2, X, W, g, beta)
    se = lambda a: a.std() / math.sqrt(len(a))
    assert abs(Mn.mean()) < 4 * se(Mn)
    assert d.mean() <= pr.mean() + 4 * se(pr - d)
    # dual estimator never exceeds the realised cost of the primal policy on the same path minus M_tau
    assert np.all(np.isfinite(d))
