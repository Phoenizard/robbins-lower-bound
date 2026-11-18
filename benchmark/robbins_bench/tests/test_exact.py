"""Independent exact / high-precision solvers: v_1..v_4 and the m = 1 column."""
import mpmath as mp
import sympy as sp

from conftest import BF_TABLE3, BF_VN
from robbins_bench import m1_exact, smalln


def test_v2():
    assert smalln.v2_exact() == sp.Rational(5, 4)


def test_v3_closed_form():
    v3, v3n, root, _ = smalln.v3_exact()
    assert sp.simplify(v3 - (sp.Rational(341, 144) - 13 * sp.sqrt(13) / 48)) == 0
    assert sp.simplify(root - (sp.Rational(5, 4) - sp.sqrt(13) / 4)) == 0
    assert str(v3n).startswith("1.39155208511739178866909979561")
    # BF print V_3 = 1.3915 (true 1.391552..., rounds to 1.3916): one unit in the last digit, documented
    assert abs(float(v3n) - float(BF_VN[3])) < 1e-4


def test_v3_mpmath_pipeline_agrees_with_sympy():
    v3n = smalln.v3_exact()[1]
    with mp.workdps(30):
        v, _ = smalln.v3_highprec(30)
        assert abs(v - mp.mpf(str(v3n))) < mp.mpf(10) ** -25


def test_v4_converged_and_matches_literature():
    with mp.workdps(30):
        a, sa = smalln.v4_highprec(20)
        b, sb = smalln.v4_highprec(30)
        assert abs(a - b) < mp.mpf(10) ** -11
        assert abs(sa - sb) < mp.mpf(10) ** -15
        # Dendievel & Swan (2016) / Brice et al. (2026) quote v_4 = 1.4932...
        assert mp.nstr(b, 12).startswith("1.4932975933")


def test_m1_column_matches_bf_except_last_digit_at_17():
    w = m1_exact.W1_values(17, 60)
    for n, s in BF_TABLE3[1].items():
        got = f"{float(w[n - 1]):.4f}"
        if n == 17:
            assert got == "1.4386" and s == "1.4385"      # documented 1-ulp discrepancy
        else:
            assert got == s


def test_m1_solver_precision_stable():
    a = m1_exact.W1_values(17, 50)
    b = m1_exact.W1_values(17, 80)
    with mp.workdps(80):
        for x, y in zip(a, b):
            assert abs(x - y) < mp.mpf(10) ** -35
