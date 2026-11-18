"""Slow reference implementation of the discretised lower/upper recursions of truncated_dp.py in
mpmath interval arithmetic (mpmath.iv), written independently and literally (direct sums over cells,
no prefix sums, dictionary storage).  Used only by the test-suite on tiny instances to check
  (a) the prefix-sum bookkeeping of the fast implementation, and
  (b) that its directed-rounding results are valid one-sided bounds of the exact discrete recursion.

Each V_k(y) is carried as an interval enclosing the EXACT real value of the discrete recursion.
int_cell min(kx + j, C) dx is nondecreasing in C, so for an interval C = [c1, c2] it is enclosed by
[F(c1).a, F(c2).b] with F evaluated in interval arithmetic at the point endpoints.
"""
from itertools import combinations_with_replacement

from mpmath import iv, mpf


def _F(c, a, b, k, j):
    """interval enclosure of int_a^b min(k x + j, c) dx for a point c (mpf), exact a, b."""
    A, B, C = iv.mpf(a), iv.mpf(b), iv.mpf(c)
    w = B - A
    if k == 0:
        return (C if c < j else iv.mpf(j)) * w
    lo_line = k * a + j        # exact rationals representable in binary (dyadic a)
    hi_line = k * b + j
    if c <= lo_line:
        return C * w
    if c >= hi_line:
        return w * (j + k * (A + B) / 2)
    D = C - (k * A + j)
    return C * w - D * D / (2 * k)


def _Fint(Civ, a, b, k, j):
    lo = _F(mpf(Civ.a), a, b, k, j)
    hi = _F(mpf(Civ.b), a, b, k, j)
    return iv.mpf([lo.a, hi.b])


def iv_enclose(m, t, nmax, lower, prec=53):
    """Return list of mpmath intervals enclosing the exact discrete L (lower=True) or U values of
    1 + V_n(1..1), n = 1..nmax."""
    old = iv.prec
    iv.prec = prec
    try:
        t = [float(x) for x in t]
        G = len(t) - 1
        tuples = list(combinations_with_replacement(range(G + 1), m))
        V = {z: iv.mpf(m) for z in tuples}
        out = []
        for k in range(nmax):
            newV = {}
            for y in tuples:
                total = (1 - iv.mpf(t[y[-1]])) * V[y]
                for j in range(m):
                    low = y[j - 1] if j > 0 else 0
                    up = y[j]
                    for c in range(low, up):          # cell (t_c, t_{c+1}) inside (y_j, y_{j+1})
                        ins = c + 1 if lower else c
                        z = tuple(y[:j]) + (ins,) + tuple(y[j:m - 1])
                        total = total + _Fint(V[z], t[c], t[c + 1], k, j)
                newV[y] = total
            V = newV
            out.append(1 + V[tuple([G] * m)])
        return out
    finally:
        iv.prec = old
