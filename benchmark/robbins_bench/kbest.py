"""Independent re-implementation of the (d,k)-abstraction M_{n,d,k} of
Brice, Bruss, Majumdar, Raskin, "Algorithms for Robbins' Problem using Markov Decision Processes",
arXiv 2608.27419v1, Definition 5 / eq. (10), and of the authors' public code (github anirban11/Robbins_MDP,
bottom-up.py), which differ in the "forgotten draws" correction term.

Model.  [0,1) is cut into d boxes [i/d,(i+1)/d).  Only the multiset of boxes of the k best (smallest) draws
seen so far is remembered (ties inside a box are averaged).  r = number of draws still to come INCLUDING the
upcoming one (code convention; the paper's r is the number left AFTER the current draw, r_paper = r - 1).
Before the upcoming draw the memory h has size s = min(k, n - r).  The draw lands in box m w.p. 1/d.
x = #{h < m}, c = #{h == m}, maxi = max(h).

Stopping loss, buffer not full (n - r < k):   1 + x + c/2 + (r-1)(m+1/2)/d ;  memory -> h + {m}.
Buffer full (n - r >= k), variant "code" (bottom-up.py), F = n-r-k-1 for r >= 2 and F = n-k-1 for r = 1:
    m <  maxi : 1 + x + c/2                                   ; memory -> drop_max(h + {m})
    m == maxi : 1 + x + c/2 + F / (2(d-m))                    ; memory -> h
    m >  maxi : 1 + k + F (m-maxi)/(d-maxi) + F/(2(d-maxi))   ; memory -> h
    plus (r-1)(m+1/2)/d in every case.
Variant "paper" (Definition 5, eq. (10)), y = drop_max(h + {m}), maxi_y = max(y), r_p = r - 1:
    m <  maxi_y : sum_{i<m} y_i + (y_m+1)/2
    m == maxi_y : sum_{i<m} y_i + (y_m+1)/2 + max(0, n-r_p-k)/(2(d-m))
    m >  maxi_y : 1 + k + max(0, n-r_p-k-1)(2m-2maxi_y+1)/(2(d-maxi_y))
    plus r_p (2m+1)/(2d);  memory -> y.
(The paper's transition rule literally writes y'[m]+1, y'[maxi]-1 also for m > maxi; we use drop-the-max,
which is what the authors' code does and the only reading under which the case m > maxi can occur.)

These are NOT rigorous bounds on v_n (the forgotten-draw terms are heuristic); this module only reproduces
the published numbers.  Plain float64 arithmetic.
"""
import math

import numba as nb
import numpy as np

from .truncated_dp import advance, binom_table, rank_of


@nb.njit
def _gen_tuples(s, G, N):
    T = np.zeros((N, max(s, 1)), dtype=np.int64)
    z = np.zeros(max(s, 1), dtype=np.int64)
    for idx in range(N):
        for q in range(s):
            T[idx, q] = z[q]
        if s > 0:
            advance(z, s, G)
    return T


@nb.njit(parallel=True)
def _level(T, s, Vnext, s_next, n, d, k, r, B, variant, last):
    N = T.shape[0]
    out = np.empty(N)
    full = (n - r) >= k
    for idx in nb.prange(N):
        h = T[idx]
        y = np.empty(k + 1, dtype=np.int64)
        maxi = h[s - 1] if s > 0 else -1
        total = 0.0
        p = 0
        for m in range(d):
            while p < s and h[p] < m:
                p += 1
            x = p
            c = 0
            q = p
            while q < s and h[q] == m:
                q += 1
                c += 1
            # memory after the draw: insert m, drop the max if size exceeds k
            ny = 0
            ins = False
            for i in range(s):
                if (not ins) and m < h[i]:
                    y[ny] = m
                    ny += 1
                    ins = True
                y[ny] = h[i]
                ny += 1
            if not ins:
                y[ny] = m
                ny += 1
            if ny > k:
                ny = k
            fut = (r - 1) * (m + 0.5) / d
            if not full:
                stop = 1.0 + x + 0.5 * c + fut
            elif variant == 0:
                F = (n - k - 1) if r == 1 else (n - r - k - 1)
                if m < maxi:
                    stop = 1.0 + x + 0.5 * c
                elif m == maxi:
                    stop = 1.0 + x + 0.5 * c + F / (2.0 * (d - m))
                else:
                    stop = 1.0 + k + F * (m - maxi) / (d - maxi) + F / (2.0 * (d - maxi))
                stop += fut
            else:
                rp = r - 1
                maxy = y[ny - 1]
                below = 0
                ym = 0
                for i in range(ny):
                    if y[i] < m:
                        below += 1
                    elif y[i] == m:
                        ym += 1
                if m < maxy:
                    stop = below + 0.5 * (ym + 1)
                elif m == maxy:
                    stop = below + 0.5 * (ym + 1) + max(0, n - rp - k) / (2.0 * (d - m))
                else:
                    stop = 1.0 + k + max(0, n - rp - k - 1) * (2 * m - 2 * maxy + 1) / (2.0 * (d - maxy))
                stop += fut
            if last:
                val = stop
            else:
                cont = Vnext[rank_of(y, s_next, B)]
                val = stop if stop < cont else cont
            total += val
        out[idx] = total / d
    return out


def kbest_value(n, d, k, variant="code"):
    """Optimal expected loss of M_{n,d,k} (variant 'code' = authors' bottom-up.py, 'paper' = Def. 5)."""
    if n == 1:
        return 1.0
    var = 0 if variant == "code" else 1
    G = d - 1
    B = binom_table(G + k + 2, k + 1)
    V = None
    s_prev = None
    for r in range(1, n + 1):
        s = min(k, n - r)
        N = math.comb(G + s, s)
        T = _gen_tuples(s, G, N)
        if r == 1:
            V = _level(T, s, np.zeros(1), 0, n, d, k, r, B, var, True)
        else:
            V = _level(T, s, V, s_prev, n, d, k, r, B, var, False)
        s_prev = s
    return float(V[0])
