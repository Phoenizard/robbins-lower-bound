"""k=3 ("best-three memory") approximate continuation-value function for Robbins' problem.

State after t observations: (a1, a2, a3) = three smallest values seen (sentinel 1 for missing).
Same recursion as dp2 with four regions:
    (0,a1): r=1, new (x,a1,a2);  (a1,a2): r=2, new (a1,x,a2);  (a2,a3): r=3, new (a1,a2,x);
    (a3,1): Markov rank rbar(x) = 4 + (s-4)(x-a3)/(1-a3), state unchanged.
W returned with shape (n, G, G, G), fully symmetrised (W[perm(i,l,k)] = W[sorted]).
"""
import numpy as np

from .grid import make_grid, default_beta
from .dp2 import panel_min_integral


def _sym_index(G):
    I = np.arange(G)
    ii, ll, kk = np.meshgrid(I, I, I, indexing="ij")
    st = np.sort(np.stack([ii, ll, kk]), axis=0)
    return st[0], st[1], st[2]


def solve_dp3(n: int, G: int, beta: float | None = None, dtype=np.float64):
    if beta is None:
        beta = default_beta(n)
    g = make_grid(G, beta)
    h = np.diff(g)
    W = np.empty((n, G, G, G), dtype=dtype)
    si, sl, sk = _sym_index(G)
    I = np.arange(G)
    Wnext = None
    for t in range(n - 1, -1, -1):
        s = t + 1
        m = n - s
        # region A: PA[j, i, l] = panel j of int min(1+mx, W(x, g_i, g_l)), j < i
        c0 = 1.0 + m * g[:-1]
        c1 = 1.0 + m * g[1:]
        if Wnext is None:
            PA = np.broadcast_to((0.5 * h * (c0 + c1))[:, None, None], (G - 1, G, G))
        else:
            PA = panel_min_integral(h[:, None, None], c0[:, None, None], c1[:, None, None],
                                    Wnext[:-1, :, :], Wnext[1:, :, :])
        CA = np.concatenate([np.zeros((1, G, G)), np.cumsum(PA, axis=0)], axis=0)
        A = CA[I, I, :]  # A[i, l] = sum_{j<i} PA[j, i, l]
        del PA, CA
        # region B: PB[i, j, l], panel j in [i, l), state (g_i, x, g_l)
        c0 = 2.0 + m * g[:-1]
        c1 = 2.0 + m * g[1:]
        if Wnext is None:
            PB = np.broadcast_to((0.5 * h * (c0 + c1))[None, :, None], (G, G - 1, G))
        else:
            PB = panel_min_integral(h[None, :, None], c0[None, :, None], c1[None, :, None],
                                    Wnext[:, :-1, :], Wnext[:, 1:, :])
        CB = np.concatenate([np.zeros((G, 1, G)), np.cumsum(PB, axis=1)], axis=1)  # CB[i,k,l]=sum_{j<k}
        B = CB[:, I, I] - CB[I, I, :]  # B[i,l] = CB[i,l,l] - CB[i,i,l]
        del PB, CB
        # region C: PC[i, l, j], panel j in [l, k), state (g_i, g_l, x)
        c0 = 3.0 + m * g[:-1]
        c1 = 3.0 + m * g[1:]
        if Wnext is None:
            PC = np.broadcast_to((0.5 * h * (c0 + c1))[None, None, :], (G, G, G - 1))
        else:
            PC = panel_min_integral(h[None, None, :], c0[None, None, :], c1[None, None, :],
                                    Wnext[:, :, :-1], Wnext[:, :, 1:])
        CC = np.concatenate([np.zeros((G, G, 1)), np.cumsum(PC, axis=2)], axis=2)
        Cr = CC - CC[:, I, I][:, :, None]  # C[i,l,k] = CC[i,l,k] - CC[i,l,l]
        del PC, CC
        # region D: (a3, 1), Markov rank
        a3 = g[None, None, :]
        hD = 1.0 - a3
        kappa = np.where(hD > 0, max(s - 4, 0) / np.where(hD > 0, hD, 1.0), 0.0)
        d0 = 4.0 + m * a3
        d1 = 4.0 + m * 1.0 + kappa * hD
        if Wnext is None:
            D = np.broadcast_to(0.5 * hD * (d0 + d1), (G, G, G))
        else:
            D = panel_min_integral(hD, d0, d1, Wnext, Wnext)
        Wt = A[:, :, None] + B[:, :, None] + Cr + D
        Wt = Wt[si, sl, sk]  # symmetrise from the sorted (i<=l<=k) entries
        W[t] = Wt
        Wnext = W[t].astype(np.float64)
    return g, W


def dp3_value(n: int, G: int, beta: float | None = None) -> float:
    g, W = solve_dp3(n, G, beta)
    return float(W[0][G - 1, G - 1, G - 1])
