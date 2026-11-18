"""Exact-rational reference implementation of the hybrid relaxation (tests only; tiny n).

Deliberately independent of encoding.py / kernel.py: for uncapped designs it keeps the summary in terms of
per-bin counts (K, N_t, N^1, ..., N^D) (not cumulative), recomputes the relaxed ranks, eq. (5), directly from
the definitions of Section 3.2, and evaluates the cell integral, eq. (6), by the three-branch formula in
fractions.Fraction (exact arithmetic, so the rounding pitfall of Remark 4.5 does not arise).  For capped
designs the caps are applied to the cumulative counts S_b = N_t + N^1 + ... + N^{b-1}, as in Lemma 3.5:
the stored summary is then (K, min(S_1, C_1), ..., min(S_D, C_D)).
"""
from fractions import Fraction as Fr
from functools import lru_cache


def panel3(lo, hi, r, m, w):
    """Cell integral, eq. (6): int_lo^hi min(r + m x, w) dx, w = None meaning +infinity (forced stop)."""
    clo, chi = r + m * lo, r + m * hi
    if w is None or w >= chi:
        return r * (hi - lo) + Fr(1, 2) * m * (hi * hi - lo * lo)
    if w <= clo:
        return w * (hi - lo)
    xs = (w - r) / m
    return r * (xs - lo) + Fr(1, 2) * m * (xs * xs - lo * lo) + w * (hi - xs)


def solve_exact(n, k, f, E, caps=None):
    f = [Fr(x) for x in f]          # Fraction(float) is the exact binary value
    E = [Fr(x) for x in E]
    Gf, D = len(f) - 1, len(E) - 1
    EMPTY = Gf

    if caps is None:
        # per-bin counts N = (N^0, ..., N^D)
        @lru_cache(maxsize=None)
        def U(t, K, N):
            s = t + 1
            m = n - s
            acc = Fr(0)
            for l in range(Gf):                     # fine cell l: (f[l], f[l+1]]
                r = 1 + sum(1 for q in K if q < l)
                w = None
                if s < n:
                    K2 = tuple(sorted(K + (l,))[:k])
                    N2 = list(N); N2[0] += 1
                    w = U(t + 1, K2, tuple(N2))
                acc += panel3(f[l], f[l + 1], r, m, w)
            for b in range(1, D + 1):               # coarse bin b: (E[b-1], E[b]]
                r = 1 + sum(N[:b])                  # N^0 + N^1 + ... + N^{b-1}
                w = None
                if s < n:
                    N2 = list(N); N2[b] += 1
                    w = U(t + 1, K, tuple(N2))
                acc += panel3(E[b - 1], E[b], r, m, w)
            return acc
        return U(0, (EMPTY,) * k, (0,) * (D + 1))

    caps = list(caps)

    @lru_cache(maxsize=None)
    def V(t, K, S):                                 # S = capped cumulative counts (S_1..S_D)
        s = t + 1
        m = n - s
        acc = Fr(0)
        for l in range(Gf):
            r = 1 + sum(1 for q in K if q < l)
            w = None
            if s < n:
                K2 = tuple(sorted(K + (l,))[:k])
                S2 = tuple(min(S[p] + 1, caps[p]) for p in range(D))
                w = V(t + 1, K2, S2)
            acc += panel3(f[l], f[l + 1], r, m, w)
        for b in range(1, D + 1):
            r = 1 + S[b - 1]
            w = None
            if s < n:
                S2 = tuple(min(S[p] + 1, caps[p]) if p >= b else S[p] for p in range(D))
                w = V(t + 1, K, S2)
            acc += panel3(E[b - 1], E[b], r, m, w)
        return acc
    return V(0, (EMPTY,) * k, (0,) * D)
