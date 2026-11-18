"""Independent replication -- exact rational reference, written from the definitions (Section 3, Lemma 3.5).

State = the FULL history as a sorted tuple of cell indices (no K/S summary, no caps recursion,
no block encoding).  The relaxed rank is recomputed from the raw history every time:
  fine cell l (0-based)  : r = 1 + #{q in K : q < l},  K = the k smallest fine indices in the history
  coarse bin b (1-based) : r = 1 + min(S_b, C_b),     S_b = #{h in history : h is fine or a coarse bin < b}
Panel: P = int_lo^hi min(r + m x, w) dx with the clamp form, exactly in Fraction.
U_{n-1}(H) = sum_c h_c r_c (forced stop, m = 0);  U_t(H) = sum_c P(lo,hi,r,n-t-1,U_{t+1}(H+c)).
Edges are the exact rationals equal to the binary64 edges.
Feasible only for tiny designs (number of histories = C(G+t-1, t)).
"""
from fractions import Fraction
from functools import lru_cache


def panel_exact(lo, hi, r, m, w):
    if m == 0:
        return min(Fraction(r), w) * (hi - lo) if w is not None else Fraction(r) * (hi - lo)
    a = (w - r) / m
    if a < lo:
        a = lo
    if a > hi:
        a = hi
    return r * (a - lo) + Fraction(m, 2) * (a * a - lo * lo) + w * (hi - a)


def exact_U0(design):
    n, k, Gf, D = design.n, design.k, design.Gf, design.D
    cells = [(Fraction(lo), Fraction(hi)) for lo, hi in design.cells()]
    caps = design.caps
    G = Gf + D

    def rank(H, c):
        if c < Gf:
            fine = sorted(h for h in H if h < Gf)[:k]
            return 1 + sum(1 for q in fine if q < c)
        b = c - Gf + 1
        S_b = sum(1 for h in H if h < Gf + b - 1)
        return 1 + min(S_b, caps[b - 1])

    @lru_cache(maxsize=None)
    def U(H):
        t = len(H)
        if t == n - 1:
            return sum(((hi - lo) * rank(H, c) for c, (lo, hi) in enumerate(cells)), Fraction(0))
        m = n - t - 1
        tot = Fraction(0)
        for c, (lo, hi) in enumerate(cells):
            w = U(tuple(sorted(H + (c,))))
            tot += panel_exact(lo, hi, rank(H, c), m, w)
        return tot

    if n == 1:
        return sum(((hi - lo) * 1 for lo, hi in cells), Fraction(0))
    return U(())
