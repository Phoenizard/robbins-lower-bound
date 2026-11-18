"""Directed-rounding primitives and grids."""
import random

import mpmath as mp
import numpy as np
import pytest

from robbins_bench.grid import SCALE, check_grid, make_grid
from robbins_bench.truncated_dp import f_hi, f_lo, f_nearest


def exact_F(C, a, b, k, j):
    """int_a^b min(kx+j, C) dx with 120-digit arithmetic (C, a, b floats taken exactly)."""
    with mp.workdps(120):
        C, a, b = mp.mpf(C), mp.mpf(a), mp.mpf(b)
        if k == 0:
            return min(C, j) * (b - a)
        xs = (C - j) / k
        if xs <= a:
            return C * (b - a)
        if xs >= b:
            return j * (b - a) + k * (b * b - a * a) / 2
        return j * (xs - a) + k * (xs * xs - a * a) / 2 + C * (b - xs)


@pytest.mark.parametrize("kind,p", [("uniform", 1.0), ("power", 1.5), ("power", 2.0)])
@pytest.mark.parametrize("G", [1, 7, 64, 1000, 2 ** 16])
def test_grid_dyadic_and_increasing(kind, p, G):
    if p * 16 > 30 and G == 2 ** 16:
        with pytest.raises(ValueError):       # first cell below the 2^-30 resolution: refused, not silently merged
            make_grid(G, kind, p)
        return
    t = make_grid(G, kind, p)
    assert check_grid(t)
    assert len(t) == G + 1
    # affine data used by the DP are exact
    for k in (0, 1, 16, 100):
        for j in (0, 3, 5):
            v = k * t + j
            assert np.all(v * SCALE == np.round(v * SCALE))


def test_directed_F_brackets_exact():
    rng = random.Random(12345)
    t = make_grid(97, "power", 1.5)
    nfail = 0
    for _ in range(4000):
        c = rng.randrange(97)
        a, b = float(t[c]), float(t[c + 1])
        k = rng.randrange(0, 18)
        j = rng.randrange(0, 6)
        mode = rng.random()
        if mode < 0.2:                   # exactly on a branch boundary
            C = k * a + j if rng.random() < 0.5 else k * b + j
        elif mode < 0.6:                 # inside the kink region
            C = k * a + j + rng.random() * k * (b - a)
        else:
            C = rng.uniform(0, 6)
        ex = exact_F(C, a, b, k, j)
        lo, hi, mid = f_lo(C, a, b, k, j), f_hi(C, a, b, k, j), f_nearest(C, a, b, k, j)
        if not (mp.mpf(lo) <= ex <= mp.mpf(hi)):
            nfail += 1
        assert abs(mid - float(ex)) <= 1e-14
        assert hi - lo <= 1e-14
    assert nfail == 0
