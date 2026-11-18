"""Exact / high-precision optimal values v_n of Robbins' problem for n = 1..4.

v_n = W_n^(n-1) (with m = n-1 the truncation in (4.4) is void), so we evaluate recursion (4.5) of
Bruss-Ferguson with m = n - 1 directly:
    v_1 = 1,   v_2 = 5/4                         (closed form)
    v_3                                         exact closed form with sympy (piecewise quadratics + one root)
    v_4                                         high precision: the two inner integrals are done exactly
                                                (closed forms + exact piece-wise polynomial integration),
                                                the outer one by adaptive Gauss-Legendre in mpmath.
These are independent of truncated_dp.py (no grid).
"""
import mpmath as mp
import sympy as sp


# ---------------------------------------------------------------------------------------------
# n = 2, 3 exactly (sympy)
# ---------------------------------------------------------------------------------------------
def v2_exact():
    x = sp.symbols("x", nonnegative=True)
    # stop on X1 = x: 1 + x ; continue: E rank of X2 = 2 - x ; threshold 1/2
    return sp.nsimplify(sp.integrate(1 + x, (x, 0, sp.Rational(1, 2))) + sp.integrate(2 - x, (x, sp.Rational(1, 2), 1)))


def _int_min_lin(lo, hi, p, c, x):
    """int_lo^hi min(x + p, c - x) dx as a sympy expression valid when the crossing (c-p)/2 lies in [lo, hi]."""
    xs = (c - p) / 2
    return sp.integrate(x + p, (x, lo, xs)) + sp.integrate(c - x, (x, xs, hi))


def v3_exact():
    """Return (exact sympy number, its 50-digit value).

    m = 2, V_1(y1, y2) = 2 - y1 - y2.
    V_2(a, 1) = int_0^a min(u, 2 - u - a) du + int_a^1 min(u + 1, 2 - a - u) du
    v_3 = 1 + int_0^1 min(2a, V_2(a, 1)) da.
    """
    a, u = sp.symbols("a u", nonnegative=True)
    # first integral: crossing u = 1 - a/2 >= a  iff a <= 2/3
    I0_small = sp.integrate(u, (u, 0, a))
    I0_large = sp.integrate(u, (u, 0, 1 - a / 2)) + sp.integrate(2 - u - a, (u, 1 - a / 2, a))
    # second integral on [a, 1]: crossing u = (1 - a)/2 >= a iff a <= 1/3
    I1_small = sp.integrate(u + 1, (u, a, (1 - a) / 2)) + sp.integrate(2 - a - u, (u, (1 - a) / 2, 1))
    I1_large = sp.integrate(2 - a - u, (u, a, 1))
    V2 = sp.Piecewise((sp.expand(I0_small + I1_small), a <= sp.Rational(1, 3)),
                      (sp.expand(I0_small + I1_large), a <= sp.Rational(2, 3)),
                      (sp.expand(I0_large + I1_large), True))
    pieces = [(0, sp.Rational(1, 3), sp.expand(I0_small + I1_small)),
              (sp.Rational(1, 3), sp.Rational(2, 3), sp.expand(I0_small + I1_large)),
              (sp.Rational(2, 3), 1, sp.expand(I0_large + I1_large))]
    # crossing 2a = V_2(a): g = 2a - V2 increasing (V_2 nonincreasing)
    root = None
    for lo, hi, P in pieces:
        g = sp.expand(2 * a - P)
        if g.subs(a, lo) <= 0 <= g.subs(a, hi):
            sols = [s for s in sp.solve(sp.Eq(g, 0), a) if s.is_real and lo <= s <= hi]
            assert len(sols) == 1
            root = (lo, hi, sp.nsimplify(sols[0]))
            break
    assert root is not None
    total = 0
    for lo, hi, P in pieces:
        if hi <= root[2]:
            total += sp.integrate(2 * a, (a, lo, hi))
        elif lo >= root[2]:
            total += sp.integrate(P, (a, lo, hi))
        else:
            total += sp.integrate(2 * a, (a, lo, root[2])) + sp.integrate(P, (a, root[2], hi))
    v3 = sp.nsimplify(sp.simplify(1 + total))
    return v3, sp.N(v3, 50), root[2], V2


# ---------------------------------------------------------------------------------------------
# n = 4 high precision (mpmath)
# ---------------------------------------------------------------------------------------------
def _J(lo, hi, p, c):
    """int_lo^hi min(x + p, c - x) dx, exact closed form, lo <= hi."""
    if hi <= lo:
        return mp.mpf(0)
    xs = (c - p) / 2
    if xs <= lo:
        return c * (hi - lo) - (hi * hi - lo * lo) / 2
    if xs >= hi:
        return p * (hi - lo) + (hi * hi - lo * lo) / 2
    return p * (xs - lo) + (xs * xs - lo * lo) / 2 + c * (hi - xs) - (hi * hi - xs * xs) / 2


def _V2(p, q):
    """V_2(p, q, 1) for n = 4 (m = 3, two observations to go), 0 <= p <= q <= 1."""
    s = 3 - p - q
    return _J(0, p, 0, s) + _J(p, q, 1, s) + _J(q, 1, 2, s)


def _int_min_poly2(f_line_p, f_line_slope, g, lo, hi, kinks):
    """int_lo^hi min(slope*x + p, g(x)) dx where g is a quadratic polynomial between consecutive kinks.

    Exact up to working precision: on each smooth piece we recover the quadratic of h = line - g from three
    point values, solve it, and integrate each sub-piece by Simpson's rule (exact for polynomials of degree
    <= 3)."""
    pts = sorted(set([lo, hi] + [k for k in kinks if lo < k < hi]))
    total = mp.mpf(0)
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a <= 0:
            continue
        m = (a + b) / 2
        h = lambda x: f_line_slope * x + f_line_p - g(x)
        ha, hm, hb = h(a), h(m), h(b)
        # quadratic through (a,ha),(m,hm),(b,hb) in t = (x - a)/(b - a)
        # h = A t^2 + B t + C
        C = ha
        A = 2 * hb - 4 * hm + 2 * ha
        Bc = hb - ha - A
        roots = []
        if abs(A) > mp.mpf(10) ** (-mp.mp.dps + 10):
            disc = Bc * Bc - 4 * A * C
            if disc >= 0:
                sq = mp.sqrt(disc)
                roots = [(-Bc - sq) / (2 * A), (-Bc + sq) / (2 * A)]
        elif abs(Bc) > 0:
            roots = [-C / Bc]
        sub = sorted(set([mp.mpf(0), mp.mpf(1)] + [r for r in roots if 0 < r < 1]))
        for t0, t1 in zip(sub[:-1], sub[1:]):
            x0 = a + (b - a) * t0
            x1 = a + (b - a) * t1
            xm = (x0 + x1) / 2
            F = lambda x: min(f_line_slope * x + f_line_p, g(x))
            total += (x1 - x0) * (F(x0) + 4 * F(xm) + F(x1)) / 6
    return total


def _V3(a):
    """V_3(a, 1, 1) for n = 4: the first observation was a, three observations to go."""
    # j = 0: new x < a, state (x, a, 1), stop cost 2x ; kinks of V_2(x, a) in x
    q = a
    k0 = [(3 - q) / 3, (2 - q) / 3, 2 - 3 * q, 1 - 3 * q]
    I0 = _int_min_poly2(0, 2, lambda x: _V2(x, a), mp.mpf(0), a, k0)
    # j = 1: new x > a, state (a, x, 1), stop cost 2x + 1 ; kinks of V_2(a, x) in x
    p = a
    k1 = [3 - 3 * p, 2 - 3 * p, (2 - p) / 3, (1 - p) / 3]
    I1 = _int_min_poly2(1, 2, lambda x: _V2(a, x), a, mp.mpf(1), k1)
    return I0 + I1


def _gl(f, n, lo, hi):
    xs, ws = _gl_nodes(n)
    h = (hi - lo) / 2
    c = (hi + lo) / 2
    return h * mp.fsum(w * f(c + h * x) for x, w in zip(xs, ws))


def _gl_adaptive(f, a, b, tol, depth=0, maxdepth=60, whole=None):
    """Adaptive 20-point Gauss-Legendre: accept an interval when the rule on the whole interval agrees with
    the composite rule on its two halves.  (Comparing two rule orders on the same interval is NOT safe here:
    at an interior kink both can agree to many digits while sharing an O(h^2) error.)"""
    if whole is None:
        whole = _gl(f, 20, a, b)
    m = (a + b) / 2
    left = _gl(f, 20, a, m)
    right = _gl(f, 20, m, b)
    if abs(whole - (left + right)) < tol or depth >= maxdepth:
        return left + right
    return (_gl_adaptive(f, a, m, tol / 2, depth + 1, maxdepth, left)
            + _gl_adaptive(f, m, b, tol / 2, depth + 1, maxdepth, right))


_GL_CACHE = {}


def _gl_nodes(n):
    key = (n, mp.mp.dps)
    if key not in _GL_CACHE:
        xs = []
        ws = []
        for i in range(1, n + 1):
            x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
            for _ in range(100):
                p0, p1 = mp.mpf(1), x
                for k in range(2, n + 1):
                    p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                dp = n * (x * p1 - p0) / (x * x - 1)
                dx = p1 / dp
                x -= dx
                if abs(dx) < mp.mpf(10) ** (-mp.mp.dps + 3):
                    break
            p0, p1 = mp.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            xs.append(x)
            ws.append(2 / ((1 - x * x) * dp * dp))
        _GL_CACHE[key] = (xs, ws)
    return _GL_CACHE[key]


def v4_highprec(dps=30, tol=None):
    """v_4 = 1 + int_0^1 min(3a, V_3(a, 1, 1)) da.  Returns (value, crossing point a*)."""
    with mp.workdps(dps):
        if tol is None:
            tol = mp.mpf(10) ** (-(dps - 8))
        g = lambda a: 3 * a - _V3(a)
        lo, hi = mp.mpf(0), mp.mpf(1)
        assert g(lo) < 0 < g(hi)
        for _ in range(int(3.4 * dps) + 10):
            mid = (lo + hi) / 2
            if g(mid) < 0:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2
        part1 = 3 * s * s / 2
        part2 = _gl_adaptive(_V3, s, mp.mpf(1), tol)
        return 1 + part1 + part2, s


def v3_highprec(dps=30, tol=None):
    """Same pipeline as v4_highprec for n = 3 (cross-check of the mpmath pipeline against sympy)."""
    with mp.workdps(dps):
        if tol is None:
            tol = mp.mpf(10) ** (-(dps - 8))

        def V2(a):   # V_2(a, 1) for n = 3 (m = 2): j=0: int_0^a min(u, 2-u-a), j=1: int_a^1 min(u+1, 2-a-u)
            return _J(0, a, 0, 2 - a) + _J(a, mp.mpf(1), 1, 2 - a)

        g = lambda a: 2 * a - V2(a)
        lo, hi = mp.mpf(0), mp.mpf(1)
        for _ in range(int(3.4 * dps) + 10):
            mid = (lo + hi) / 2
            if g(mid) < 0:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2
        return 1 + s * s + _gl_adaptive(V2, s, mp.mpf(1), tol), s
