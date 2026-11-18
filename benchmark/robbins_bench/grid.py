"""Exact dyadic grids 0 = t_0 < t_1 < ... < t_G = 1.

Every grid point is an integer multiple of 2**-SHIFT_BITS (SHIFT_BITS = 30), so that for integers
0 <= k <= 2**20 and 0 <= j <= 64 the quantities k*t_i + j, t_{i+1} - t_i, t_i + t_{i+1}, j*(t_{i+1}-t_i)
and 1 - t_i are computed EXACTLY in IEEE double precision.  The rigorous DP relies on this.
"""
import numpy as np

SHIFT_BITS = 30
SCALE = float(2 ** SHIFT_BITS)


def _to_dyadic(u):
    q = np.round(np.asarray(u, dtype=np.float64) * SCALE) / SCALE
    return q


def make_grid(G, kind="power", p=1.5):
    """Return float64 array t of length G+1 with exact dyadic entries.

    kind = "uniform": t_i = i/G
    kind = "power"  : t_i = (i/G)**p   (finer near 0, where stopping decisions concentrate)
    """
    if G < 1:
        raise ValueError("G >= 1 required")
    u = np.arange(G + 1, dtype=np.float64) / G
    if kind == "uniform":
        t = u
    elif kind == "power":
        t = u ** p
    else:
        raise ValueError(kind)
    t = _to_dyadic(t)
    t[0] = 0.0
    t[-1] = 1.0
    if not np.all(np.diff(t) > 0):
        raise ValueError("grid not strictly increasing after dyadic rounding; reduce G or p")
    check_grid(t)
    return t


def check_grid(t):
    t = np.asarray(t, dtype=np.float64)
    assert t[0] == 0.0 and t[-1] == 1.0
    assert np.all(np.diff(t) > 0)
    # every entry is a multiple of 2^-SHIFT_BITS
    s = t * SCALE
    assert np.all(s == np.round(s)), "grid points must be dyadic with <= 30 fractional bits"
    return True
