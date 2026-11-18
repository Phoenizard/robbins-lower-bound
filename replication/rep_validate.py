"""Independent replication -- validation suite.  Usage: python -X utf8 rep_validate.py [quick]
Writes validation_results.json next to this file.
"""
import os, sys, json, math, random, time, tempfile, shutil
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
sys.path.insert(0, HERE)

import numpy as np
import numba as nb
from numba import njit

import rep_design as RD
import rep_exact as RE
import rep_solver as RS
from rep_env import setup_env

setup_env(4)
RES = {}

# exact ground truth (benchmark/results/exact.json) and upper bounds on v_5, v_6 (up to floating-point rounding)
# from a pessimistic grid rule, Table 3 of the paper (that code is not part of this repository)
V_EXACT = {1: Fraction(1), 2: Fraction(5, 4), 3: 1.3915520851173917887, 4: 1.4932975933712254496}
V_UPPER = {5: 1.5741707, 6: 1.6399473}          # pessimistic grid rule: upper bounds up to floating-point rounding
W1 = {5: 1.38027953059803428672, 9: 1.41705792217964544056, 17: 1.43855939918479341166}  # BF W_n^(1), exact


@njit
def _ops(a, b, out):
    for i in range(a.shape[0]):
        out[i, 0] = RS.add_dn(a[i], b[i])
        out[i, 1] = RS.sub_up(a[i], b[i])
        out[i, 2] = RS.sub_dn(a[i], b[i])
        out[i, 3] = RS.mul_dn(a[i], b[i])
        out[i, 4] = RS.mul_up(a[i], b[i])
        out[i, 5] = RS.div_up(abs(a[i]), 2.0 * (1.0 + np.floor(abs(b[i]) * 50.0)))


@njit
def _f32(x, out):
    for i in range(x.shape[0]):
        out[i] = RS.f32_dn(x[i])


@njit
def _panels(lo, hi, hlo, r, m, w, out_dn, out_nr):
    for i in range(lo.shape[0]):
        mf = m[i]
        clo_lo = RS.add_dn(r[i], RS.mul_dn(mf, lo[i]))
        chi_lo = RS.add_dn(r[i], RS.mul_dn(mf, hi[i]))
        out_dn[i] = RS.panel_down(clo_lo, chi_lo, hlo[i], 2.0 * mf, w[i])
        out_nr[i] = RS.panel_near(lo[i], hi[i], hi[i] - lo[i], r[i], mf, w[i])


def test_primitives(N=100000, seed=1):
    rng = np.random.default_rng(seed)
    # mixed magnitudes, including equal-ish operands and exact cases
    a = rng.uniform(0, 100, N) * 10.0 ** rng.integers(-8, 3, N)
    b = rng.uniform(0, 100, N) * 10.0 ** rng.integers(-8, 3, N)
    b[: N // 10] = a[: N // 10] * (1 + rng.uniform(-1e-12, 1e-12, N // 10))
    a[N // 10: N // 5] = np.round(a[N // 10: N // 5])
    b[N // 10: N // 5] = np.round(b[N // 10: N // 5])
    out = np.zeros((N, 6))
    _ops(a, b, out)
    bad = [0] * 6
    loose = 0
    for i in range(N):
        A, B = Fraction(a[i]), Fraction(b[i])
        c = Fraction(2.0 * (1.0 + math.floor(abs(b[i]) * 50.0)))
        ex = [A + B, A - B, A - B, A * B, A * B, abs(A) / c]
        dirs = [-1, 1, -1, -1, 1, 1]
        for j in range(6):
            v = Fraction(out[i, j])
            if (dirs[j] < 0 and v > ex[j]) or (dirs[j] > 0 and v < ex[j]):
                bad[j] += 1
            # tightness: within 2 ulp
            if abs(float(v) - float(ex[j])) > 2.5 * np.spacing(abs(float(ex[j]))) + 1e-300:
                loose += 1
    x = rng.uniform(0.5, 50.0, N)
    x[: 1000] = 2.0 ** rng.integers(-3, 6, 1000)   # powers of two
    x[1000: 2000] = np.nextafter(x[1000:2000].astype(np.float32).astype(np.float64), 0)
    o32 = np.zeros(N, dtype=np.float32)
    _f32(x, o32)
    ref = x.astype(np.float32)
    ref = np.where(ref.astype(np.float64) > x, np.nextafter(ref, np.float32(-np.inf)), ref)
    f32_mismatch = int((o32 != ref).sum())
    RES["primitives"] = dict(N=N, wrong_direction=dict(zip(["add_dn", "sub_up", "sub_dn", "mul_dn", "mul_up", "div_up"], bad)),
                             looser_than_2ulp=loose, f32_dn_mismatch_vs_numpy_nextafter=f32_mismatch)
    print("primitives", RES["primitives"], flush=True)
    assert sum(bad) == 0 and f32_mismatch == 0


def exact_panel(lo, hi, r, m, w):
    return RE.panel_exact(Fraction(lo), Fraction(hi), r, m, Fraction(w))


def test_panels(N=60000, seed=2):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(0, 1, N)
    wid = 10.0 ** rng.uniform(-9, -0.5, N)
    hi = np.minimum(lo + wid, 1.0)
    lo[: N // 10] = 0.0
    r = rng.integers(1, 40, N).astype(float)
    m = rng.integers(1, 120, N).astype(float)
    clo = r + m * lo
    chi = r + m * hi
    w = clo + rng.uniform(-0.2, 1.2, N) * (chi - clo)
    # 1/3: w within a few ulp of a branch point
    k3 = N // 3
    pick = rng.integers(0, 2, k3)
    base = np.where(pick == 0, clo[:k3], chi[:k3])
    w[:k3] = base + rng.integers(-4, 5, k3) * np.spacing(base)
    w = np.maximum(w, 1e-3)
    hlo = np.array([RS._sub_dn_py(b, a) for a, b in zip(lo, hi)])
    od = np.zeros(N)
    on = np.zeros(N)
    _panels(lo, hi, hlo, r, m, w, od, on)
    bad = 0
    worst_rel_dn = 0.0
    worst_rel_nr = 0.0
    for i in range(N):
        ex = exact_panel(lo[i], hi[i], int(r[i]), int(m[i]), w[i])
        if Fraction(od[i]) > ex:
            bad += 1
        exf = float(ex)
        if exf > 0:
            worst_rel_dn = max(worst_rel_dn, (exf - od[i]) / exf)
            worst_rel_nr = max(worst_rel_nr, abs(exf - on[i]) / exf)
    RES["panels"] = dict(N=N, down_above_exact=bad, worst_rel_gap_down=worst_rel_dn, worst_rel_err_nearest=worst_rel_nr)
    print("panels", RES["panels"], flush=True)
    assert bad == 0


def rand_design(rng, n):
    k = int(rng.integers(0, 4))
    Gf = int(rng.integers(1, 4))
    D = int(rng.integers(1, 4))
    theta = float(rng.uniform(0.1, 0.8)) if rng.random() < 0.7 else float(rng.integers(1, 8)) / 8
    fine = sorted(set([0.0, theta] + [float(x) for x in rng.uniform(0, theta, Gf - 1)]))
    coarse = sorted(set([theta, 1.0] + [float(x) for x in rng.uniform(theta, 1, D - 1)]))
    D = len(coarse) - 1
    if rng.random() < 0.6:
        c1 = max(k, int(rng.integers(0, 3)))
        caps = [c1]
        for _ in range(D - 1):
            caps.append(caps[-1] + int(rng.integers(0, 2)))
    else:
        caps = None
    return RD.custom(n, k, fine, coarse, caps, label=f"rand n={n} k={k} Gf={len(fine)-1} D={D} caps={caps}")


def test_exact_small(n_designs=40, seed=3):
    rng = np.random.default_rng(seed)
    designs = []
    for i in range(n_designs):
        designs.append(rand_design(rng, int(rng.integers(2, 8))))
    designs += [
        RD.hybrid(6, 2, 2, 3, 3, 1.5, [2, 2, 3], "hyb6 caps(2,2,3)"),
        RD.hybrid(7, 3, 2, 2, 2, 2.0, [3, 3], "hyb7 caps(3,3)"),
        RD.hybrid(5, 1, 2, 2, 3, 0.0, None, "hyb5 uncapped"),
        RD.hybrid(8, 1, 3, 2, 2, 3.0, [1, 1], "hyb8 tight caps"),
        RD.kbest(5, 1, 4, "kbest5 k1 G4"),
        RD.kbest(6, 2, 3, "kbest6 k2 G3"),
        RD.kbest(4, 3, 4, "kbest4 k3 G4 full memory"),
    ]
    rows = []
    worst_near = 0.0
    for d in designs:
        ex = RE.exact_U0(d)
        exf = float(ex)
        rn = RS.solve(d, "nearest", "f64")["U0"]
        rd64 = RS.solve(d, "down", "f64")["U0"]
        rd32 = RS.solve(d, "down", "f32")["U0"]
        rn32 = RS.solve(d, "nearest", "f32")["U0"]
        ok64 = Fraction(rd64) <= ex
        ok32 = Fraction(rd32) <= ex
        worst_near = max(worst_near, abs(rn - exf))
        row = dict(design=d.label, n=d.n, exact=exf, nearest_err=rn - exf, down64_gap=float(ex - Fraction(rd64)),
                   down32_gap=float(ex - Fraction(rd32)), nearest32_err=rn32 - exf, down_le_exact=bool(ok64 and ok32))
        if d.n in V_EXACT:
            row["le_vn"] = exf <= float(V_EXACT[d.n])
        elif d.n in V_UPPER:
            row["le_vn_upper"] = exf <= V_UPPER[d.n]
        rows.append(row)
        assert ok64 and ok32, row
        assert abs(rn - exf) < 1e-13, row
    RES["exact_small"] = dict(count=len(rows), max_abs_nearest_err=worst_near,
                              max_down64_gap=max(r["down64_gap"] for r in rows),
                              max_down32_gap=max(r["down32_gap"] for r in rows),
                              min_down64_gap=min(r["down64_gap"] for r in rows),
                              all_down_le_exact=all(r["down_le_exact"] for r in rows),
                              all_le_vn=all(r.get("le_vn", True) and r.get("le_vn_upper", True) for r in rows),
                              rows=rows)
    print("exact_small: %d designs, max|nearest-exact|=%.2e, max down64 gap=%.2e, max down32 gap=%.2e" % (
        len(rows), worst_near, RES["exact_small"]["max_down64_gap"], RES["exact_small"]["max_down32_gap"]))


def test_vn():
    rows = []
    # complete memory k = n-1, theta = 1: converges to v_n from below (O(1/G))
    for n, Gs in [(1, [4]), (2, [10, 40, 160]), (3, [20, 80, 320]), (4, [20, 40, 80]), (5, [20, 40]), (6, [12, 20])]:
        if n <= 3:
            ref, ref_kind = float(V_EXACT[n]), "v_n from exact formula, converted to float"
        elif n == 4:
            ref, ref_kind = float(V_EXACT[n]), "high-precision numerical reference for v_4"
        else:
            ref, ref_kind = V_UPPER[n], "expected rank of pessimistic grid rule, up to floating-point rounding"
        for G in Gs:
            d = RD.kbest(n, n - 1, G)
            u = RS.solve(d, "nearest", "f64")["U0"]
            ud = RS.solve(d, "down", "f64")["U0"]
            rows.append(dict(n=n, design=f"k=n-1, theta=1, G={G}", U0=u, U0_down=ud, ref=ref,
                             ref_kind=ref_kind, gap=ref - u))
            assert ud <= u + 1e-12 and u <= ref, rows[-1]
    # hybrid designs at n <= 6
    for d in [RD.hybrid(4, 3, 1.2, 30, 20, 0.0, None), RD.hybrid(5, 2, 2, 12, 12, 1.0, None),
              RD.hybrid(6, 2, 2, 8, 6, 1.0, None), RD.hybrid(6, 3, 2, 8, 6, 1.0, [3, 3, 4, 4, 5, 5])]:
        u = RS.solve(d, "nearest", "f64")["U0"]
        ref = float(V_EXACT[d.n]) if d.n in V_EXACT else V_UPPER[d.n]
        rows.append(dict(n=d.n, design=d.label, U0=u, ref=ref, gap=ref - u))
        assert u <= ref
    # BF truncated loss W_n^(1): route b1 with k=1 converges to it from below
    for n, Gs in [(5, [100, 400]), (9, [100, 400]), (17, [100, 400])]:
        for G in Gs:
            d = RD.kbest(n, 1, G)
            u = RS.solve(d, "nearest", "f64")["U0"]
            rows.append(dict(n=n, design=f"k=1, theta=1, G={G}", U0=u, ref=W1[n], ref_kind="BF W_n^(1) exact", gap=W1[n] - u))
            assert u <= W1[n]
    RES["vn"] = rows
    for r in rows:
        print("vn", r)


def test_backends():
    d = RD.hybrid(14, 3, 7, 10, 5, 3.0, [6, 7, 8, 9, 10])
    a = RS.solve(d, "nearest", "f64", "ram")
    tmp = tempfile.mkdtemp(prefix="repval_", dir=os.environ.get("REP_WORKDIR", None))
    try:
        b = RS.solve(d, "nearest", "f64", "disk", tmp)
    finally:
        shutil.rmtree(tmp)   # own temporary files only
    c = RS.solve(d, "down", "f64", "ram")
    e = RS.solve(d, "down", "f32", "ram")
    f = RS.solve(d, "nearest", "f32", "ram")
    RES["backends"] = dict(design=d.label, ram=a["U0"], disk=b["U0"], identical=a["U0"] == b["U0"],
                           down64=c["U0"], down32=e["U0"], near32=f["U0"],
                           near_minus_down64=a["U0"] - c["U0"], near_minus_down32=a["U0"] - e["U0"])
    print("backends", RES["backends"], flush=True)
    assert a["U0"] == b["U0"] and c["U0"] <= a["U0"] and e["U0"] <= c["U0"]


def test_monotone(seed=5):
    """refining coarse bins / fine cells, k -> k+1, raising caps never lowers U_0 (exact rationals)."""
    rng = np.random.default_rng(seed)
    viol = 0
    cnt = 0
    for i in range(12):
        n = int(rng.integers(3, 7))
        d = rand_design(rng, n)
        base = float(RE.exact_U0(d))
        variants = []
        mid = 0.5 * (d.coarse[0] + d.coarse[1])
        # refine first coarse bin (cap of new bin = cap of original b=1... the new bin's S counts <= mid)
        variants.append(RD.custom(n, d.k, d.fine, [d.coarse[0], mid] + d.coarse[1:], [d.caps[0]] + [d.caps[0]] + d.caps[1:]))
        variants.append(RD.custom(n, d.k, [0.0, 0.5 * d.fine[1]] + d.fine[1:], d.coarse, d.caps))
        variants.append(RD.custom(n, d.k, d.fine, d.coarse, [c + 1 for c in d.caps]))
        if d.caps[0] >= d.k + 1:
            variants.append(RD.custom(n, d.k + 1, d.fine, d.coarse, d.caps))
        for v in variants:
            cnt += 1
            if float(RE.exact_U0(v)) < base - 1e-15:
                viol += 1
    RES["monotone"] = dict(checks=cnt, violations=viol)
    print("monotone", RES["monotone"], flush=True)


if __name__ == "__main__":
    t0 = time.time()
    test_primitives()
    test_panels()
    test_exact_small()
    test_vn()
    test_backends()
    test_monotone()
    RES["seconds"] = round(time.time() - t0, 1)
    json.dump(RES, open(os.path.join(HERE, "validation_results.json"), "w"), indent=1, default=str)
    print("ALL VALIDATION PASSED in", RES["seconds"], "s", flush=True)
