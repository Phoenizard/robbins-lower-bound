"""The rigorous discretised DP for W_n^(m): consistency with the interval reference, with independent
solvers, and with Bruss-Ferguson Table 3 (small m, moderate grids; the production grids are checked in
test_results.py)."""
import mpmath as mp
import pytest

from conftest import BF_TABLE3
from robbins_bench import m1_exact, smalln
from robbins_bench.grid import make_grid
from robbins_bench.ivref import iv_enclose
from robbins_bench.truncated_dp import enclose_W, enclose_W_both


@pytest.mark.parametrize("m,G,n", [(1, 8, 6), (2, 6, 5), (3, 5, 5), (4, 4, 4), (2, 5, 4)])
@pytest.mark.parametrize("lower", [True, False])
def test_fast_dp_vs_mpmath_iv_reference(m, G, n, lower):
    """Fast numba DP (prefix sums + directed rounding) is a valid one-sided bound of the exact discrete
    recursion, as enclosed by the literal mpmath.iv implementation, and agrees with it to 1e-12."""
    for grid in (make_grid(G, "power", 1.5), make_grid(G, "uniform")):
        fast = enclose_W(m, grid, n, lower)
        ref = iv_enclose(m, grid, n, lower)
        for a, b in zip(fast, ref):
            if lower:
                assert a <= float(b.b)
                assert a >= float(b.a) - 1e-12
            else:
                assert a >= float(b.a)
                assert a <= float(b.b) + 1e-12


def test_trivial_values():
    for m in (1, 2, 3):
        lo, hi = enclose_W_both(m, make_grid(16, "power", 1.5), 2)
        assert lo[0] <= 1.0 <= hi[0]          # W_1 = 1
        assert lo[1] <= 1.25 <= hi[1]         # W_2^(m) = v_2 = 5/4 for all m >= 1


def test_m1_bracket_contains_independent_solver():
    ex = m1_exact.W1_values(17, 50)
    lo, hi = enclose_W_both(1, make_grid(4096, "power", 1.5), 17)
    for n in range(1, 18):
        assert lo[n - 1] <= float(ex[n - 1]) <= hi[n - 1]
        assert hi[n - 1] - lo[n - 1] < 3e-4


def test_full_information_brackets_contain_v3_v4():
    v3 = float(smalln.v3_exact()[1])
    lo, hi = enclose_W_both(2, make_grid(1024, "power", 1.5), 3)
    assert lo[2] <= v3 <= hi[2] and hi[2] - lo[2] < 3e-3
    v4, _ = smalln.v4_highprec(20)
    lo, hi = enclose_W_both(3, make_grid(96, "power", 1.5), 4)
    assert lo[3] <= float(v4) <= hi[3]


def test_bracket_shrinks_with_grid():
    w = []
    for G in (32, 128, 512):
        lo, hi = enclose_W_both(2, make_grid(G, "power", 1.5), 9)
        w.append(hi[8] - lo[8])
    assert w[0] > w[1] > w[2]
    assert w[2] < 0.25 * w[0]


def test_monotone_in_m_and_n():
    """W_n^(m) is nondecreasing in m (BF: payoff nondecreasing in m) and in n (BF Theorem 2)."""
    g = make_grid(48, "power", 1.5)
    br = {m: enclose_W_both(m, g, 9) for m in (1, 2, 3)}
    for m in (1, 2):
        for n in range(1, 10):
            assert br[m][0][n - 1] <= br[m + 1][1][n - 1]
    for m in (1, 2, 3):
        for n in range(1, 9):
            assert br[m][0][n - 1] <= br[m][1][n]


@pytest.mark.parametrize("m,G", [(1, 4096), (2, 512), (3, 128)])
def test_bf_table3_small_m_moderate_grid(m, G):
    """BF Table 3 entries lie inside rigorous enclosures already at moderate grids."""
    lo, hi = enclose_W_both(m, make_grid(G, "power", 1.5), 17)
    for n, s in BF_TABLE3[m].items():
        v = float(s)
        assert lo[n - 1] - 5e-5 <= v <= hi[n - 1] + 5e-5, (m, n, lo[n - 1], v, hi[n - 1])


def test_documented_mismatch_W17_m1():
    """BF print W_17^(1) = 1.4385; the true value is 1.438559... (independent exact solver and a rigorous
    enclosure agree), which rounds to 1.4386.  Last-digit discrepancy, documented in README."""
    ex = m1_exact.W1_values(17, 50)[16]
    assert mp.nstr(ex, 6) == "1.43856"
    lo, hi = enclose_W_both(1, make_grid(2 ** 16, "power", 1.5), 17)
    assert lo[16] > 1.43855 and hi[16] < 1.43857
