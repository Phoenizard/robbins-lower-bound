"""Monte Carlo check of W_n^(m) directly from the DEFINITION (4.4), independent of recursion (4.5).

For ANY stopping rule tau adapted to X_1, X_2, ...,  E[Y_tau] >= W_n^(m) = inf_N E[Y_N], where
    Y_k = 1 + min{m, #{i<k: X_i < X_k}} + (n-k) X_k.
We use the (heuristic) rule "stop iff Y_k - 1 < V_{n-k}(state after k, rounded down to the grid)" with V taken
from a discrete value table, simulate it with exact uniform variables, and report mean +- standard error.
So W_n^(m) <= E[Y_tau] and, with high confidence, W_n^(m) < mean + 3 SE: an upper check on BF's numbers
that does not depend on our reading of (4.5).  Fixed seeds; single thread.
"""
import numba as nb
import numpy as np

from .truncated_dp import binom_table, rank_of


@nb.njit
def _simulate(Vflat, offsets, m, n, t, B, npaths, seed):
    np.random.seed(seed)
    G = t.shape[0] - 1
    y = np.empty(m)
    z = np.empty(m, dtype=np.int64)
    ynew = np.empty(m)
    s1 = 0.0
    s2 = 0.0
    for _ in range(npaths):
        for i in range(m):
            y[i] = 1.0
        loss = 0.0
        for k in range(1, n + 1):
            x = np.random.random()
            cnt = 0
            for i in range(m):
                if y[i] < x:
                    cnt += 1
            togo = n - k
            stopcost = cnt + togo * x            # = Y_k - 1
            # state after observing x
            if cnt < m:
                p = 0
                for i in range(m):
                    if i == cnt:
                        ynew[p] = x
                        p += 1
                    if p < m:
                        ynew[p] = y[i]
                        p += 1
                    if p >= m:
                        break
                # (if cnt == m - 1 the loop above places x last)
            else:
                for i in range(m):
                    ynew[i] = y[i]
            if k == n:
                loss = 1.0 + stopcost
                break
            for i in range(m):
                # grid index: largest g with t[g] <= ynew[i]
                lo, hi = 0, G
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if t[mid] <= ynew[i]:
                        lo = mid
                    else:
                        hi = mid - 1
                z[i] = lo
            cont = Vflat[offsets[togo] + rank_of(z, m, B)]
            if stopcost < cont:
                loss = 1.0 + stopcost
                break
            for i in range(m):
                y[i] = ynew[i]
        s1 += loss
        s2 += loss * loss
    mean = s1 / npaths
    var = s2 / npaths - mean * mean
    return mean, np.sqrt(var / npaths)


def simulate_policy(m, n, t, tables, npaths, seed):
    """tables = [V_0, ..., V_{n-1}] (at least) from truncated_dp.value_tables."""
    t = np.ascontiguousarray(t, dtype=np.float64)
    G = len(t) - 1
    B = binom_table(G + m + 1, m)
    sizes = [len(tables[k]) for k in range(n)]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    Vflat = np.concatenate([tables[k] for k in range(n)])
    return _simulate(Vflat, offsets, m, n, t, B, npaths, seed)
