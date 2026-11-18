"""k=2 ("best-two memory") approximate continuation-value function for Robbins' problem.

State after t observations: (a1, a2) = the two smallest values seen (a2 = 1 sentinel if t < 2,
a1 = a2 = 1 if t = 0).  W[t][i, l] approximates the optimal expected cost of *continuing* after
t observations when (a1, a2) = (g_i, g_l), i <= l.  Arrival s = t + 1 with value x costs
    c_s(x) = r_s(x) + (n - s) x,      r_s = relative rank of x among the first s values,
and stopping is compulsory at s = n.

Recursion (exact integration of the piecewise-linear interpolant of W[s] along grid lines):
    W[t](a1,a2) = int_0^a1   min(1 + (n-s)x, W[s](x, a1)) dx
                + int_a1^a2  min(2 + (n-s)x, W[s](a1, x)) dx
                + int_a2^1   min(rbar(x) + (n-s)x, W[s](a1, a2)) dx
with the *Markov approximation* rbar(x) = 3 + (s-3)(x-a2)/(1-a2) for x > a2 (the s-3 forgotten
values treated as iid U(a2,1)).  This approximation only affects the quality of W as a value
function; the dual bound in dual.py is valid for any W.
W[n] = +inf (forced stop).  The approximate value is W[0](1,1).

W is returned symmetrised (W[l,i] = W[i,l]) so that bilinear interpolation is defined everywhere.
"""
import numpy as np

from .grid import make_grid, default_beta


def panel_min_integral(h, c0, c1, w0, w1):
    """Exact integral over a panel of width h of min(c, w), c and w linear with end values."""
    d0 = c0 - w0
    d1 = c1 - w1
    base = 0.5 * h * (c0 + c1)
    both_pos = (d0 >= 0) & (d1 >= 0)
    both_neg = (d0 <= 0) & (d1 <= 0)
    denom = np.abs(d0) + np.abs(d1)
    denom = np.where(denom > 0, denom, 1.0)
    mixed = 0.5 * h * np.maximum(d0, d1) ** 2 / denom
    pos = np.where(both_pos, 0.5 * h * (d0 + d1), np.where(both_neg, 0.0, mixed))
    return base - pos


def solve_dp2(n: int, G: int, beta: float | None = None):
    """Return (g, W) with W of shape (n, G, G); W[t] = continuation value after t observations."""
    if beta is None:
        beta = default_beta(n)
    g = make_grid(G, beta)
    W = np.empty((n, G, G))
    Wnext = None  # W[n] = +inf
    I = np.arange(G)
    for t in range(n - 1, -1, -1):
        s = t + 1
        m = n - s
        # ---- region A: x in (g_j, g_{j+1}) below a1 = g_i, new state (x, g_i); r = 1 -------------
        # PA[j, i] for j < i
        cA0 = 1.0 + m * g[:-1]          # at x = g_j
        cA1 = 1.0 + m * g[1:]           # at x = g_{j+1}
        h = np.diff(g)
        if Wnext is None:
            PA = (0.5 * h * (cA0 + cA1))[:, None] * np.ones((1, G))
        else:
            PA = panel_min_integral(h[:, None], cA0[:, None], cA1[:, None], Wnext[:-1, :], Wnext[1:, :])
        # A[i] = sum_{j < i} PA[j, i]
        CA = np.vstack([np.zeros((1, G)), np.cumsum(PA, axis=0)])  # CA[k, i] = sum_{j<k} PA[j,i]
        A = CA[I, I]
        # ---- region B: x in (a1, a2), new state (g_i, x); r = 2 ------------------------------
        cB0 = 2.0 + m * g[:-1]
        cB1 = 2.0 + m * g[1:]
        if Wnext is None:
            PB = np.ones((G, 1)) * (0.5 * h * (cB0 + cB1))[None, :]
        else:
            PB = panel_min_integral(h[None, :], cB0[None, :], cB1[None, :], Wnext[:, :-1], Wnext[:, 1:])
        CB = np.hstack([np.zeros((G, 1)), np.cumsum(PB, axis=1)])  # CB[i, k] = sum_{j<k} PB[i,j]
        B = CB - CB[I, I][:, None]  # B[i, l] = sum_{i<=j<l} PB[i, j]
        # ---- region C: x in (a2, 1), state unchanged; Markov-approximate rank -----------------
        a2 = g[None, :] * np.ones((G, 1))
        hC = 1.0 - a2
        kappa = np.where(hC > 0, max(s - 3, 0) / np.where(hC > 0, hC, 1.0), 0.0)
        c0 = 3.0 + m * a2
        c1 = 3.0 + m * 1.0 + kappa * hC
        if Wnext is None:
            C = 0.5 * hC * (c0 + c1)
        else:
            C = panel_min_integral(hC, c0, c1, Wnext, Wnext)
        Wt = A[:, None] + B + C
        # symmetrise: keep upper triangle (i <= l)
        Wt = np.triu(Wt) + np.triu(Wt, 1).T
        W[t] = Wt
        Wnext = Wt
    return g, W


def dp2_value(n: int, G: int, beta: float | None = None) -> float:
    g, W = solve_dp2(n, G, beta)
    return float(W[0][G - 1, G - 1])
