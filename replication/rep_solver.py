"""Independent replication -- backward DP of the capped hybrid relaxation (Lemmas 3.1, 3.4, 3.5, 4.2).

Written from the written specification of the relaxation and of the sign-aware rounding only, without
access to the main implementation.  Does not import robbins_lb.

State after t observations: (K, s) with
  s = (s_1..s_D), s_b = min(#{obs <= E_{b-1}}, C_b)    (nondecreasing; reachable iff s_D <= t)
  K = multiset of the j = min(k, s_1) smallest fine-cell indices  (C_1 >= k makes j exact)
Encoding (deliberately not lexicographic ranking):
  * all capped vectors s are enumerated once, sorted by (s_D, s_{D-1}, ..., s_1); layer t is the
    PREFIX {s_D <= t}; a dense mixed-radix lookup table is used once to build successor tables;
  * K is ranked inside its block j by a dict built from itertools (block offsets koff[j]);
  * flat state index = off[g] + kidx, identical in every layer -> layers are prefixes of one index space;
  * "slab" s = {g : s_D = s} is a contiguous range; successors of slab s lie in slabs s, s+1,
    so a layer is computed slab by slab with only 3 slabs in RAM (out-of-core option).
Arithmetic:
  nearest : P = y h - max(0, y - clo)^2 / (2m), y = min(w, chi), plain IEEE.
  down    : the same form with independently implemented directed rounding: each +,-,*,/ is done in
            round-to-nearest and its exact error sign is obtained by an error-free transformation
            (TwoSum / Dekker TwoProduct); only if the result is on the wrong side is it moved by
            >= 1 ulp (x -/+ |x| 2^-52).  f32 storage in down mode uses a frexp-based float32 predecessor.
Summation order over cells: coarse bins D..1 first, then fine cells Gf..1 (reverse of the natural order).
"""
import itertools
import math
import os
import time
import json

import numpy as np
import numba as nb
from numba import njit, prange

# ---------------------------------------------------------------- directed rounding primitives
_TWO_M52 = 2.0 ** -52
_SPLIT = 134217729.0  # 2^27 + 1 (Veltkamp)


@njit(inline="always", cache=False)
def step_dn(x):
    ax = abs(x)
    if ax > 1e-290:
        return x - ax * _TWO_M52
    return np.nextafter(x, -np.inf)


@njit(inline="always", cache=False)
def step_up(x):
    ax = abs(x)
    if ax > 1e-290:
        return x + ax * _TWO_M52
    return np.nextafter(x, np.inf)


@njit(inline="always", cache=False)
def two_sum_err(a, b, s):
    bb = s - a
    return (a - (s - bb)) + (b - bb)


@njit(inline="always", cache=False)
def add_dn(a, b):
    s = a + b
    if two_sum_err(a, b, s) < 0.0:
        return step_dn(s)
    return s


@njit(inline="always", cache=False)
def sub_dn(a, b):
    return add_dn(a, -b)


@njit(inline="always", cache=False)
def sub_up(a, b):
    s = a - b
    if two_sum_err(a, -b, s) > 0.0:
        return step_up(s)
    return s


@njit(inline="always", cache=False)
def two_prod_err(a, b, p):
    # Dekker: exact a*b = p + e (no overflow/underflow in our ranges)
    ca = _SPLIT * a
    ah = ca - (ca - a)
    al = a - ah
    cb = _SPLIT * b
    bh = cb - (cb - b)
    bl = b - bh
    return ((ah * bh - p) + ah * bl + al * bh) + al * bl


@njit(inline="always", cache=False)
def mul_dn(a, b):
    p = a * b
    if two_prod_err(a, b, p) < 0.0:
        return step_dn(p)
    return p


@njit(inline="always", cache=False)
def mul_up(a, b):
    p = a * b
    if two_prod_err(a, b, p) > 0.0:
        return step_up(p)
    return p


@njit(inline="always", cache=False)
def div_up(a, c):
    """upper bound of a / c for a >= 0, c > 0."""
    q = a / c
    p = q * c
    e = two_prod_err(q, c, p)
    resid = (p - a) + e          # sign of q*c - a ; p - a exact (Sterbenz)
    if resid < 0.0:
        return step_up(q)
    return q


@njit(inline="always", cache=False)
def f32_dn(x):
    """largest float32 <= x (x > 0 normal), via frexp; does not use nextafter."""
    v = np.float32(x)
    if np.float64(v) > x:
        fv = np.float64(v)
        mnt, ex = math.frexp(fv)          # fv = mnt * 2^ex, mnt in [0.5, 1)
        if mnt == 0.5:
            fv = fv - math.ldexp(1.0, ex - 25)
        else:
            fv = fv - math.ldexp(1.0, ex - 24)
        v = np.float32(fv)
    return v


# ---------------------------------------------------------------- panels
@njit(inline="always", cache=False)
def panel_near(lo, hi, h, r, m, w):
    clo = r + m * lo
    chi = r + m * hi
    y = w if w < chi else chi
    d = y - clo
    if d <= 0.0:
        return y * h
    return y * h - d * d / (2.0 * m)


@njit(inline="always", cache=False)
def panel_down(clo_lo, chi_lo, h_lo, m2, w):
    """lower bound of int_lo^hi min(r + m x, w) dx.
    clo_lo <= r + m lo, chi_lo <= r + m hi, h_lo <= hi - lo, m2 = 2m exact, w > 0.
    Uses P(w) >= P(y) for y = min(w, chi_lo) <= chi, and P(y) = y h - max(0, y - clo)^2/(2m)."""
    y = w if w < chi_lo else chi_lo
    yh = mul_dn(y, h_lo)
    diff = sub_up(y, clo_lo)
    if diff <= 0.0:
        return yh
    d2 = mul_up(diff, diff)
    q = div_up(d2, m2)
    return sub_dn(yh, q)


# ---------------------------------------------------------------- layer kernels
def _make_kernel(down, closed):
    @njit(parallel=True, cache=False)
    def kern(glo, ghi, S, jblk, off, succ, koff, cnt, nxt, k, Gf, D,
             lo, hi, h, h_lo, m,
             nextA, baseA, nextB, baseB, A, B,
             out, out_base, out_f32):
        mf = float(m)
        m2 = 2.0 * mf
        for g in prange(glo, ghi):
            j = jblk[g]
            jn = j + 1 if j < k else k
            nk = koff[j + 1] - koff[j]
            gfine = succ[g, 0]
            ofine = off[gfine] + 0
            for kk in range(nk):
                kg = koff[j] + kk
                acc = 0.0
                # coarse bins D..1
                for b in range(D, 0, -1):
                    c = Gf + b - 1
                    r = 1.0 + S[g, b - 1]
                    gs = g if b == D else succ[g, b]
                    if closed:
                        if down:
                            w = add_dn(A[kg], B[gs])
                        else:
                            w = A[kg] + B[gs]
                    else:
                        idx = off[gs] + kk
                        if idx < baseB:
                            w = np.float64(nextA[idx - baseA])
                        else:
                            w = np.float64(nextB[idx - baseB])
                    if down:
                        clo_lo = add_dn(r, mul_dn(mf, lo[c]))
                        chi_lo = add_dn(r, mul_dn(mf, hi[c]))
                        acc = add_dn(acc, panel_down(clo_lo, chi_lo, h_lo[c], m2, w))
                    else:
                        acc += panel_near(lo[c], hi[c], h[c], r, mf, w)
                # fine cells Gf..1
                for l in range(Gf - 1, -1, -1):
                    r = 1.0 + cnt[kg, l]
                    nkk = nxt[kg, l]
                    if closed:
                        if down:
                            w = add_dn(A[koff[jn] + nkk], B[gfine])
                        else:
                            w = A[koff[jn] + nkk] + B[gfine]
                    else:
                        idx = ofine + nkk
                        if idx < baseB:
                            w = np.float64(nextA[idx - baseA])
                        else:
                            w = np.float64(nextB[idx - baseB])
                    if down:
                        clo_lo = add_dn(r, mul_dn(mf, lo[l]))
                        chi_lo = add_dn(r, mul_dn(mf, hi[l]))
                        acc = add_dn(acc, panel_down(clo_lo, chi_lo, h_lo[l], m2, w))
                    else:
                        acc += panel_near(lo[l], hi[l], h[l], r, mf, w)
                o = off[g] + kk - out_base
                if out_f32:
                    if down:
                        out[o] = f32_dn(acc)
                    else:
                        out[o] = acc
                else:
                    out[o] = acc
    return kern


_KERNELS = {}


def get_kernel(down, closed):
    key = (down, closed)
    if key not in _KERNELS:
        _KERNELS[key] = _make_kernel(down, closed)
    return _KERNELS[key]


# ---------------------------------------------------------------- tables
def k_tables(Gf, k):
    blocks = [list(itertools.combinations_with_replacement(range(Gf), j)) for j in range(k + 1)]
    index = [{K: i for i, K in enumerate(bl)} for bl in blocks]
    koff = np.zeros(k + 2, dtype=np.int64)
    for j in range(k + 1):
        koff[j + 1] = koff[j] + len(blocks[j])
    Ktot = int(koff[-1])
    cnt = np.zeros((Ktot, max(Gf, 1)), dtype=np.int32)
    nxt = np.zeros((Ktot, max(Gf, 1)), dtype=np.int64)
    for j in range(k + 1):
        jn = min(j + 1, k)
        for i, K in enumerate(blocks[j]):
            for l in range(Gf):
                cnt[koff[j] + i, l] = sum(1 for q in K if q < l)
                K2 = tuple(sorted(K + (l,)))[:jn] if jn > 0 else ()
                nxt[koff[j] + i, l] = index[jn][K2]
    return blocks, koff, cnt, nxt


def s_tables(n, D, caps):
    capn = [min(c, n - 1) for c in caps]
    # enumerate nondecreasing vectors with s_b <= capn[b]
    arr = np.arange(capn[0] + 1, dtype=np.int32).reshape(-1, 1)
    for b in range(1, D):
        last = arr[:, -1]
        reps = np.maximum(capn[b] - last + 1, 0)
        rows = np.repeat(arr, reps, axis=0)
        start = np.repeat(last, reps)
        # within each group: values last..capn[b]
        grp_start = np.cumsum(reps) - reps
        pos = np.arange(rows.shape[0]) - np.repeat(grp_start, reps)
        arr = np.concatenate([rows, (start + pos).reshape(-1, 1).astype(np.int32)], axis=1)
    # sort by (s_D, s_{D-1}, ..., s_1): lexsort uses last key as primary
    order = np.lexsort(tuple(arr[:, b] for b in range(D)))
    S = np.ascontiguousarray(arr[order])
    NS = S.shape[0]
    # mixed-radix codes; lookup by binary search in the sorted code list (used once to build succ)
    radix = [c + 1 for c in capn]
    assert float(np.prod([float(x) for x in radix])) < 2.0 ** 62
    stride = np.ones(D, dtype=np.int64)
    for b in range(D - 2, -1, -1):
        stride[b] = stride[b + 1] * radix[b + 1]
    codes = S.astype(np.int64) @ stride
    corder = np.argsort(codes)
    csorted = codes[corder]

    def lookup(c):
        pos = np.searchsorted(csorted, c)
        pos = np.minimum(pos, NS - 1)
        found = csorted[pos] == c
        return np.where(found, corder[pos], -1)
    capv = np.array(capn, dtype=np.int32)
    succ = np.zeros((NS, max(D, 1)), dtype=np.int64)
    # pattern 0: fine arrival -> all s_b += 1; pattern b (1..D-1): coarse bin b -> s_{b'} += 1 for b' >= b+1
    for p in range(D):
        inc = np.zeros(D, dtype=np.int32)
        if p == 0:
            inc[:] = 1
        else:
            inc[p:] = 1
        S2 = np.minimum(S + inc, capv)
        c2 = S2.astype(np.int64) @ stride
        ok = S2[:, -1] <= n - 1
        sp = np.full(NS, -1, dtype=np.int64)
        sp[ok] = lookup(c2[ok])
        assert (sp[ok] >= 0).all()
        succ[:, p] = sp
    return S, succ


class Model:
    def __init__(self, design):
        d = design
        self.d = d
        self.n, self.k, self.Gf, self.D = d.n, d.k, d.Gf, d.D
        self.blocks, self.koff, self.cnt, self.nxt = k_tables(self.Gf, self.k)
        self.S, self.succ = s_tables(self.n, self.D, d.caps)
        nK = np.diff(self.koff)
        self.jblk = np.minimum(self.S[:, 0], self.k).astype(np.int64)
        sizes = nK[self.jblk]
        self.off = np.zeros(self.S.shape[0] + 1, dtype=np.int64)
        np.cumsum(sizes, out=self.off[1:])
        sD = self.S[:, -1]
        self.CD = int(sD.max())
        self.slab_start = np.searchsorted(sD, np.arange(self.CD + 2)).astype(np.int64)
        cells = d.cells()
        self.lo = np.array([c[0] for c in cells])
        self.hi = np.array([c[1] for c in cells])
        self.h = self.hi - self.lo
        # rigorous lower bound of hi - lo
        self.h_lo = np.array([_sub_dn_py(b, a) for a, b in cells])

    def layer_slabs(self, t):
        return min(t, self.CD)

    def layer_size(self, t):
        g = self.slab_start[self.layer_slabs(t) + 1]
        return int(self.off[g])

    def slab_range(self, s):
        """flat index range of slab s"""
        return int(self.off[self.slab_start[s]]), int(self.off[self.slab_start[s + 1]])

    def closed_form(self, down):
        """A[K global] and B[g] with U_{n-1}(K, s) = A[K] + B[s] (forced stop)."""
        Gf, D = self.Gf, self.D
        Ktot = int(self.koff[-1])
        A = np.zeros(Ktot)
        B = np.zeros(self.S.shape[0])
        if not down:
            for l in range(Gf):
                A += self.h[l] * (1.0 + self.cnt[:, l])
            for b in range(D):
                B += self.h[Gf + b] * (1.0 + self.S[:, b])
        else:
            A, B = _closed_dn(self.cnt, self.S, self.h_lo, Gf, D, Ktot)
        return A, B


@njit(cache=False)
def _closed_dn(cnt, S, h_lo, Gf, D, Ktot):
    A = np.zeros(Ktot)
    B = np.zeros(S.shape[0])
    for i in range(Ktot):
        a = 0.0
        for l in range(Gf):
            a = add_dn(a, mul_dn(h_lo[l], 1.0 + cnt[i, l]))
        A[i] = a
    for g in range(S.shape[0]):
        bsum = 0.0
        for b in range(D):
            bsum = add_dn(bsum, mul_dn(h_lo[Gf + b], 1.0 + S[g, b]))
        B[g] = bsum
    return A, B


def _sub_dn_py(a, b):
    """rigorous lower bound of a - b in Python (Fraction check)."""
    from fractions import Fraction
    s = a - b
    if Fraction(s) > Fraction(a) - Fraction(b):
        s = np.nextafter(s, -np.inf)
    return float(s)


# ---------------------------------------------------------------- storage backends
class RamStore:
    """two parity banks of slab arrays."""
    def __init__(self):
        self.bank = [{}, {}]

    def get(self, t, s):
        return self.bank[t % 2][s]

    def put(self, t, s, arr):
        self.bank[t % 2][s] = arr

    def drop_layer(self, t):
        self.bank[t % 2] = {}

    def nbytes(self):
        return sum(a.nbytes for bk in self.bank for a in bk.values())


class DiskStore:
    """one file per (parity, slab); layer t goes to bank t%2, so a crash never corrupts layer t+1."""
    def __init__(self, path, dtype):
        self.path = path
        self.dtype = dtype
        for p in (0, 1):
            os.makedirs(os.path.join(path, f"bank{p}"), exist_ok=True)

    def fn(self, t, s):
        return os.path.join(self.path, f"bank{t % 2}", f"slab_{s:04d}.bin")

    def get(self, t, s):
        return np.fromfile(self.fn(t, s), dtype=self.dtype)

    def put(self, t, s, arr):
        tmp = self.fn(t, s) + ".tmp"
        arr.tofile(tmp)
        os.replace(tmp, self.fn(t, s))

    def drop_layer(self, t):
        pass

    def nbytes(self):
        return 0


def _key(design, mode, storage):
    return json.dumps(dict(n=design.n, k=design.k, fine=[x.hex() for x in design.fine],
                           coarse=[x.hex() for x in design.coarse], caps=design.caps,
                           mode=mode, storage=storage))


def solve(design, mode="nearest", storage="f64", backend="ram", workdir=None, log=None):
    """Backward DP; returns dict with U0 and stats.  backend='disk' keeps <= 3 slabs in RAM."""
    down = (mode == "down")
    dt = np.float64 if storage == "f64" else np.float32
    out_f32 = (storage == "f32")
    M = Model(design)
    n = M.n
    t0 = time.time()
    A, B = M.closed_form(down)
    layer_sizes = {t: M.layer_size(t) for t in range(n)}
    if n == 1:
        return dict(U0=float(A[0] + B[0]), seconds=0.0, layer_sizes=layer_sizes)
    store = RamStore() if backend == "ram" else DiskStore(workdir, dt)
    empty = np.zeros(1, dtype=dt)
    BIG = 1 << 62
    start_t = n - 2
    ck = os.path.join(workdir, "progress.json") if backend == "disk" else None
    if ck and os.path.exists(ck):
        meta = json.load(open(ck))
        if meta.get("key") == _key(design, mode, storage) and meta["done_t"] < n - 1:
            start_t = meta["done_t"] - 1
            if log:
                log(f"resume: layer {meta['done_t']} on disk, continuing at t={start_t}")
    peak = 0
    panels = 0
    for t in range(start_t, -1, -1):
        m = n - t - 1
        closed = (t == n - 2)
        kern = get_kernel(down, closed)
        ns = M.layer_slabs(t)
        ns_next = M.layer_slabs(t + 1)
        tl = time.time()
        cur = None if closed else store.get(t + 1, 0)
        for s in range(ns + 1):
            o0, o1 = M.slab_range(s)
            out = np.empty(o1 - o0, dtype=dt)
            nxt_arr = None
            if closed:
                nA, bA, nB, bB = empty, 0, empty, BIG
            else:
                nA, bA = cur, o0
                if s + 1 <= ns_next:
                    nxt_arr = store.get(t + 1, s + 1)
                    nB, bB = nxt_arr, M.slab_range(s + 1)[0]
                else:
                    nB, bB = empty, BIG
            kern(int(M.slab_start[s]), int(M.slab_start[s + 1]), M.S, M.jblk, M.off, M.succ,
                 M.koff, M.cnt, M.nxt, M.k, M.Gf, M.D,
                 M.lo, M.hi, M.h, M.h_lo, m, nA, bA, nB, bB, A, B, out, o0, out_f32)
            panels += (o1 - o0) * (M.Gf + M.D)
            store.put(t, s, out)
            if backend == "ram":
                live = store.nbytes()
            else:
                live = out.nbytes + (nA.nbytes if nA is not empty else 0) + (nB.nbytes if nB is not empty else 0)
            peak = max(peak, live)
            del out
            cur = nxt_arr
        store.drop_layer(t + 1)
        if ck:
            json.dump(dict(key=_key(design, mode, storage), done_t=t), open(ck, "w"))
        if log:
            log(f"t={t:3d} m={m:3d} states={layer_sizes[t]:,} {time.time() - tl:.1f}s")
    u0 = store.get(0, 0)[0]
    return dict(U0=float(u0), U0_hex=float(u0).hex(), seconds=round(time.time() - t0, 2),
                max_stored_layer_states=max(layer_sizes[t] for t in range(n - 1)),
                last_layer_states=layer_sizes[n - 1],
                stored_states=sum(layer_sizes[t] for t in range(n - 1)),
                all_states=sum(layer_sizes.values()), panels=panels,
                peak_layer_bytes=peak, layer_sizes=layer_sizes)
