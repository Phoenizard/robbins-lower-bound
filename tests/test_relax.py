"""Tests for robbins_lb.  Run from the repository root:  python -m pytest tests -q

Anchors: exact v_1..v_4 from benchmark/results/exact.json (computed by the independent benchmark).
"""
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from robbins_lb import Design, solve  # noqa: E402
from robbins_lb.design import geo_edges  # noqa: E402
from robbins_lb.encoding import (count_table, enumerate_layer, layer_encoding, memory_tables,  # noqa: E402
                                 next_seq, rank_seq, unrank_seq)
from robbins_lb.exact import solve_exact  # noqa: E402

EXACT = json.load(open(os.path.join(HERE, "..", "benchmark", "results", "exact.json")))
V = {1: 1.0, 2: 1.25, 3: float(EXACT["v3_50digits"]), 4: float(EXACT["v4_dps40"])}
V5_APPROX = 1.5707   # approximate v_5 from the literature (not rigorous; used only as a sanity check)
MC_RULE_30 = 2.0522  # Monte Carlo expected rank of an explicit rule at n = 30 (upper estimate of v_30, +-0.0017)


def _d(*a, **kw):
    return Design.hybrid(*a, **kw)


# ----------------------------------------------------------------------------------------- encoding
@pytest.mark.parametrize("ub", [[0], [3], [2, 2], [1, 3, 3], [2, 4, 5, 5], [3, 3, 3, 3, 3], [0, 0, 2]])
def test_rank_unrank_next(ub):
    ub = np.array(ub, dtype=np.int64)
    seqs = enumerate_layer(list(ub))
    cnt, pref, base, _ = layer_encoding(ub, 2, np.array([1, 4, 10]))
    assert pref[0, ub[0] + 1] == len(seqs)
    S = np.zeros(len(ub), dtype=np.int64)
    for i, sq in enumerate(seqs):
        assert rank_seq(np.array(sq, dtype=np.int64), pref) == i
        unrank_seq(i, cnt, S)
        assert tuple(S) == sq
    S = np.array(seqs[0], dtype=np.int64)
    for i in range(1, len(seqs)):
        assert next_seq(S, ub)
        assert tuple(S) == seqs[i]
    assert not next_seq(S, ub)


def test_count_matches_binomial_uncapped():
    for t in range(0, 9):
        for D in range(1, 6):
            cnt = count_table([t] * D)
            assert cnt[0].sum() == math.comb(t + D, D)


@pytest.mark.parametrize("k,Gf", [(0, 3), (1, 4), (2, 5), (3, 4), (4, 3)])
def test_memory_tables(k, Gf):
    mt = memory_tables(k, Gf)
    assert mt["nk"] == sum(math.comb(Gf + j - 1, j) for j in range(k + 1))
    for g, tp in enumerate(mt["tuples"]):
        j = sum(1 for q in tp if q < Gf)
        assert mt["block"][g] == j
        for l in range(Gf):
            new = mt["tuples"][mt["nxt"][g, l]]
            assert new == tuple(sorted(tp + (l,))[:k])
            assert mt["block"][mt["nxt"][g, l]] == min(k, j + 1)
            assert mt["cnt"][g, l] == sum(1 for q in tp if q < l)


def test_layer_state_count_formula():
    # uncapped: sum over v of C(t-v+D-1, D-1) * bs[min(k, v)]
    d = _d(12, 2, 4, 5, 3, 1.0)
    for t in range(12):
        ref = sum(math.comb(t - v + d.D - 1, d.D - 1) * math.comb(d.Gf + min(2, v) - 1, min(2, v))
                  for v in range(t + 1))
        assert d.layer_states(t) == ref


# ----------------------------------------------------------------------------------------- exact rationals
SMALL = [
    (3, 1, 1.5, 3, 2, 1.0, None),
    (4, 1, 2, 4, 3, 1.0, None),
    (5, 2, 2, 3, 3, 1.5, None),
    (6, 0, 2, 1, 4, 1.0, None),
    (5, 3, 3, 3, 2, 1.0, None),
    (6, 2, 2.5, 3, 3, 2.0, None),
    (5, 2, 2, 3, 3, 1.5, 3),
    (6, 1, 2, 3, 4, 1.0, (2, 2, 3, 4)),
    (6, 0, 2, 1, 4, 1.0, 2),
]


@pytest.mark.parametrize("n,k,th,Gf,D,bc,caps", SMALL)
def test_exact_rational_agreement(n, k, th, Gf, D, bc, caps):
    d = _d(n, k, th, Gf, D, bc, caps=caps)
    ex = solve_exact(n, k, d.f, d.E, caps=d.caps)
    near = solve(d, mode="nearest", threads=2)["U0"]
    down = solve(d, mode="down", threads=2)["U0"]
    down32 = solve(d, mode="down", storage="f32", threads=2)["U0"]
    assert abs(near - float(ex)) < 1e-13
    assert Fraction(down) <= ex                     # rigorous direction, compared exactly
    assert Fraction(down32) <= ex
    assert float(ex) - down < 1e-12


# ----------------------------------------------------------------------------------------- U_0 <= v_n
DESIGNS_SMALL_N = [
    dict(k=0, th=1.0, Gf=2, D=3, bc=1.0),
    dict(k=1, th=1.5, Gf=10, D=3, bc=1.0),
    dict(k=2, th=2.0, Gf=20, D=4, bc=1.5),
    dict(k=3, th=2.5, Gf=12, D=4, bc=1.5),
]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("j", range(len(DESIGNS_SMALL_N)))
@pytest.mark.parametrize("mode", ["nearest", "down"])
def test_below_exact_v(n, j, mode):
    p = DESIGNS_SMALL_N[j]
    d = _d(n, p["k"], min(p["th"], 0.75 * n), p["Gf"], p["D"], p["bc"])
    u = solve(d, mode=mode, threads=2)["U0"]
    assert u <= V[n] + (0.0 if mode == "down" else 1e-12)
    if n == 1:
        assert abs(u - 1.0) < 1e-12


def test_below_v5_approx():
    for p in DESIGNS_SMALL_N:
        d = _d(5, p["k"], p["th"], p["Gf"], p["D"], p["bc"])
        assert solve(d, threads=2)["U0"] < V5_APPROX


def _finest(n, G, D=1):
    # complete memory (k = n-1, so nothing below theta is ever forgotten), theta = 1 - 2^-8, one tiny coarse bin
    theta = 1.0 - 2.0 ** -8
    f = geo_edges(0.0, theta, G, 0.0)
    E = geo_edges(theta, 1.0, D, 0.0)
    return Design(n, n - 1, tuple(f), tuple(E))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_finest_design_approaches_v(n):
    gaps = []
    for G in [10, 20, 40, 80, 160]:
        u = solve(_finest(n, G), threads=8)["U0"]
        assert u <= V[n] + 1e-12
        gaps.append(V[n] - u)
    assert all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1))
    # grid loss is O(1/G): halving G-spacing should roughly halve the gap
    assert gaps[-1] < 0.6 * gaps[-2]
    assert gaps[-1] < 5e-3


# ----------------------------------------------------------------------------------------- rounding modes
@pytest.mark.parametrize("n,k,th,Gf,D,bc,caps", [(20, 2, 6, 10, 4, 1.5, None), (25, 1, 5, 12, 5, 2.0, None),
                                                 (20, 3, 6, 6, 4, 1.5, None), (25, 2, 6, 8, 5, 1.5, 12)])
def test_down_below_nearest(n, k, th, Gf, D, bc, caps):
    d = _d(n, k, th, Gf, D, bc, caps=caps)
    near = solve(d, mode="nearest")["U0"]
    down = solve(d, mode="down")["U0"]
    near32 = solve(d, mode="nearest", storage="f32")["U0"]
    down32 = solve(d, mode="down", storage="f32")["U0"]
    assert down <= near
    assert near - down < 1e-10
    assert down32 <= down
    assert down - down32 < n * 4e-7        # <= one float32 ulp (at values <= n+1) lost per layer
    assert abs(near32 - near) < n * 4e-7


# ----------------------------------------------------------------------------------------- monotonicity
def _with_edges(d, f=None, E=None, k=None, caps="keep"):
    return Design(d.n, d.k if k is None else k, tuple(d.f if f is None else f), tuple(d.E if E is None else E),
                  d.caps if caps == "keep" else caps)


def _refine(edges):
    e = np.asarray(edges)
    mid = (e[:-1] + e[1:]) / 2
    out = np.empty(2 * len(e) - 1)
    out[0::2] = e
    out[1::2] = mid
    return out


MONO_BASE = [_d(12, 1, 4, 5, 3, 1.5), _d(15, 2, 5, 4, 3, 2.0), _d(10, 0, 3, 1, 3, 1.0)]


@pytest.mark.parametrize("j", range(len(MONO_BASE)))
def test_monotone_coarse_refinement(j):
    d = MONO_BASE[j]
    u0 = solve(d)["U0"]
    E = np.asarray(d.E)
    # add one edge in every bin, one bin at a time, then all
    for b in range(d.D):
        E2 = np.sort(np.append(E, (E[b] + E[b + 1]) / 2))
        assert solve(_with_edges(d, E=E2))["U0"] >= u0 - 1e-13
    assert solve(_with_edges(d, E=_refine(E)))["U0"] >= u0 - 1e-13


@pytest.mark.parametrize("j", [0, 1])
def test_monotone_fine_refinement_and_k(j):
    d = MONO_BASE[j]
    u0 = solve(d)["U0"]
    assert solve(_with_edges(d, f=_refine(d.f)))["U0"] >= u0 - 1e-13
    assert solve(_with_edges(d, k=d.k + 1))["U0"] >= u0 - 1e-13


def test_monotone_caps():
    d = _d(14, 1, 4, 5, 4, 1.5)
    prev = -1.0
    for c in [1, 2, 3, 5, 8, 13]:
        u = solve(_d(14, 1, 4, 5, 4, 1.5, caps=c))["U0"]
        assert u >= prev - 1e-13
        prev = u
    assert solve(d)["U0"] >= prev - 1e-13
    # a cap >= n is no cap at all
    assert abs(solve(_d(14, 1, 4, 5, 4, 1.5, caps=14))["U0"] - solve(d)["U0"]) < 1e-14


# ----------------------------------------------------------------------------------------- regression vs earlier float runs
@pytest.mark.parametrize("n,k,th,Gf,D,bc,ref", [(30, 2, 6, 20, 5, 1.5, 1.9622), (30, 1, 5, 20, 5, 1.5, 1.9352),
                                                 (20, 2, 6, 20, 6, 1.5, 1.8999), (10, 2, 4, 30, 8, 1.0, 1.7574)])
def test_float_regression_values(n, k, th, Gf, D, bc, ref):
    u = solve(_d(n, k, th, Gf, D, bc))["U0"]
    assert abs(u - ref) < 6e-5
    assert u < MC_RULE_30


def test_checkpoint_resume(tmp_path):
    d = _d(14, 2, 5, 6, 4, 1.5)
    ref = solve(d)["U0"]
    ck = str(tmp_path / "ck")
    solve(d, ckpt_dir=ck, ckpt_every_s=-1.0)
    meta = json.load(open(os.path.join(ck, "meta.json")))
    assert meta["done"] and meta["U0"] == ref
    # pretend the run died after storing layer t=1
    meta.pop("done")
    meta["t"] = 1
    json.dump(meta, open(os.path.join(ck, "meta.json"), "w"))
    assert solve(d, ckpt_dir=ck)["U0"] == ref


# ----------------------------------------------------------------------------------------- adversarial panels
def _p_exact(lo, hi, r, m, w):
    from robbins_lb.exact import panel3
    return panel3(Fraction(lo), Fraction(hi), Fraction(r), Fraction(m), Fraction(w))


def test_panel_down_adversarial():
    """Random cells: the 'down' panel must be <= the exact rational P in 100% of 200 000 cases; a third of the
    cases put w within a few ulps of the branch points r+m*lo, r+m*hi."""
    from robbins_lb.kernel import panel_vec
    rng = np.random.default_rng(20261112)
    N = 200_000
    lo = rng.random(N)
    hi = lo + rng.random(N) ** 3 * (1 - lo)
    hi = np.where(hi > lo, hi, np.nextafter(lo, 2.0))
    r = rng.integers(1, 31, N).astype(float)
    m = rng.integers(1, 101, N).astype(float)
    clo, chi = r + m * lo, r + m * hi
    u = rng.random(N)
    w = clo + u * (chi - clo)                                  # interior (branch 3)
    sel = rng.integers(0, 6, N)
    jit = rng.integers(-3, 4, N)
    w = np.where(sel == 1, clo, w)
    w = np.where(sel == 2, chi, w)
    w = np.where(sel == 3, clo * (1 + jit * 2.0 ** -52), w)
    w = np.where(sel == 4, chi * (1 + jit * 2.0 ** -52), w)
    w = np.where(sel == 5, rng.random(N) * (chi + 2), w)       # any branch
    w = np.maximum(w, 1e-3)
    # a worst case found during the code review
    lo[0], hi[0], r[0], m[0] = 0.96428, 0.96575, 24.0, 99.0
    w[0] = (r[0] + m[0] * lo[0]) * 0.5 + (r[0] + m[0] * hi[0]) * 0.5
    dn_ = panel_vec(True, lo, hi, r, m, w)
    nr_ = panel_vec(False, lo, hi, r, m, w)
    bad = 0
    maxrel = 0.0
    for i in range(N):
        ex = _p_exact(lo[i], hi[i], r[i], m[i], w[i])
        if Fraction(dn_[i]) > ex:
            bad += 1
        exf = float(ex)
        maxrel = max(maxrel, abs(nr_[i] - exf) / exf, (exf - dn_[i]) / exf)
    assert bad == 0
    assert maxrel < 1e-11


def test_down_never_above_exact_on_random_small_designs():
    rng = np.random.default_rng(7)
    for _ in range(6):
        n = int(rng.integers(3, 7)); k = int(rng.integers(0, 3)); D = int(rng.integers(1, 4))
        Gf = int(rng.integers(1, 4))
        f = np.sort(rng.random(Gf - 1)) * 0.4 if Gf > 1 else np.array([])
        f = np.concatenate([[0.0], np.sort(f), [0.4 + 0.1 * rng.random()]])
        E = np.concatenate([[f[-1]], np.sort(f[-1] + (1 - f[-1]) * rng.random(D - 1)), [1.0]])
        if np.any(np.diff(f) <= 0) or np.any(np.diff(E) <= 0):
            continue
        d = Design(n, k, tuple(f), tuple(E))
        ex = solve_exact(n, k, d.f, d.E)
        assert Fraction(solve(d, mode="down", threads=2)["U0"]) <= ex
        assert Fraction(solve(d, mode="down", storage="f32", threads=2)["U0"]) <= ex


def test_dn_up_match_nextafter():
    from robbins_lb.kernel import dn_up_vec
    rng = np.random.default_rng(3)
    a = rng.standard_normal(1_000_000) * 10.0 ** rng.integers(-320, 308, 1_000_000)
    special = np.array([0.0, -0.0, 5e-324, -5e-324, 1e-310, -1e-310, 2.2250738585072014e-308, 1.0, -1.0, 2.0,
                        0.5, 3.0, 1.7976931348623157e308, -1.7976931348623157e308, np.inf, -np.inf])
    with np.errstate(over="ignore"):
        a = np.concatenate([special, a, np.nextafter(special, np.inf), np.nextafter(special, -np.inf)])
        a = a[np.isfinite(a) | np.isinf(a)]
        out = dn_up_vec(a)
        ref_dn = np.nextafter(a, -np.inf)
        ref_up = np.nextafter(a, np.inf)
    assert np.array_equal(out[0].view(np.int64), ref_dn.view(np.int64))
    assert np.array_equal(out[1].view(np.int64), ref_up.view(np.int64))
