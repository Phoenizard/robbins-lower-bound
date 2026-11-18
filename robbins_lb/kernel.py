"""Numba kernel: one backward Bellman layer, eq. (8), of the hybrid relaxation (Corollary 3.8 of the paper).

Cell integral.  For a cell (lo, hi] with h = hi - lo, integer rank minorant r >= 1, m = n - s >= 1 remaining
observations and continuation value w (> 0):
    P(w) := int_lo^hi min(r + m x, w) dx.
With clo = r + m lo, chi = r + m hi and g(y) := y h - (y - clo)^2 / (2m):
    P(w) = w h                 if w <= clo,
    P(w) = g(min(w, chi))      if w >= clo.
(Derivation: for w in [clo, chi], P = int_lo^{x*} (r+mx) dx + w (hi - x*) with m(x* - lo) = w - clo, which
simplifies to g(w); for w >= chi, P = h (clo+chi)/2 = g(chi).)  This is eq. (9) of the paper; it is
algebraically identical to the three-branch formula, and it is the formula used in both modes.

Facts used by the rounding-down mode (Lemma 4.1 of the paper):
  (F1) g(y) <= P(y) for every real y  (g is a concave parabola touching P on [clo, chi], below it elsewhere);
  (F2) P is nondecreasing in w;
  hence g(y) <= P(w) whenever y <= w, *whatever* y is -- so the branch point chi need not be computed exactly;
  (F3) if w <= clo then P(w) = w h; and P(w) >= P(c) = c h for every c <= min(w, clo).

------------------------------------------------------------------------------------------------------------
ROUNDING MODES (what is rounded, and how)
------------------------------------------------------------------------------------------------------------
All inputs are IEEE binary64 numbers taken as exact reals: the edges lo, hi (exact dyadic rationals), the
integers r, m, 2m (exact), and the stored continuation value w (in "down" mode w is itself a proven lower
bound for the true continuation value U_{t+1}; by (F2) a lower bound of P(w) is then a lower bound of
P(U_{t+1})).  Hardware arithmetic is IEEE round-to-nearest (numba, fastmath off, no FMA contraction).

  dn(x) := nextafter(x, -inf),   up(x) := nextafter(x, +inf)   (implemented by an exact bit-pattern
  increment/decrement, identical to np.nextafter on every float64; see dn/up below).
  If z = fl(a op b) is the round-to-nearest result of one operation op in {+,-,*,/} then
  dn(z) <= a op b <= up(z)  (the exact result lies within half an ulp of z, the neighbour is a full ulp away).
  So applying dn (resp. up) immediately after every single operation yields a lower (upper) bound of the exact
  real expression, as long as the bound is propagated monotonically (below: all multiplications are of
  nonnegative numbers, and subtractions take a lower bound minus an upper bound or vice versa).

mode="nearest": plain IEEE round-to-nearest everywhere; float exploration only (error ~ 1e-13 relative).

mode="down" (rigorous lower bound; used for the certified runs).  Per cell (Definition 4.3, Lemma 4.4):
  hd    = dn(hi - lo)                              <= h            (precomputed once per cell)
  clo_d = dn(r + dn(m*lo))                         <= clo
  if w <= clo_d:  return dn(w*hd)                  <= w h = P(w)   (F3, since w <= clo)
  clo_u = up(r + up(m*lo)),  chi_u = up(r + up(m*hi))   (>= clo, >= chi)
  y     = min(w, chi_u)                            (exact; y <= w)
  d     = max(up(y - clo_d), up(clo_u - y))        >= |y - clo|
  q     = up(up(d*d) / (2m))                       >= (y - clo)^2 / (2m)
  g_d   = dn(dn(y*hd) - q)                         <= g(y) <= P(w)    (y > 0, hd <= h; F1, F2)
  c_d   = dn(clo_d*hd)                             <= P(w)            (F3 with c = clo_d < w)
  return max(g_d, c_d)
Sum over the Gf + D cells of a state: acc = dn(acc + p) after each addition (all p > 0).
Last step (s = n, m = 0, forced stop): U_{n-1}(K,S) = sum_l h_l (1 + cnt_l(K)) + sum_b h_b (1 + S_b) exactly;
  in "down" mode A(K) and B(S) are accumulated with dn after every product and every addition, and
  w = dn(A + B).  This layer is never stored (closed form inside layer n-2).
Storage: float64 stores acc as is.  float32 storage stores v32 = float32(acc) (round to nearest) and, in
  "down" mode, replaces it by its float32 predecessor if float64(v32) > acc, so the stored number is <= acc
  (Lemma 4.2: any stored number below the computed lower bound keeps the induction valid).
Branch selection never needs to be exact: the "w <= clo_d" test only selects a formula that is valid when the
test is true, and the other branch returns a valid lower bound for every w (F1-F3).

This is the sign-aware form: the minuend y*h is rounded down, the subtrahend d, d^2, d^2/(2m) up.  The
three-branch form with an approximate x* is never used (it is the minimum of a convex function of x*, so its
rounding errors go the wrong way; Remark 4.5).  The branch need not be selected exactly (F1-F3), so dyadic
edges are not needed.
Adversarial test: tests/test_relax.py::test_panel_down_adversarial (200k random cells vs exact rationals).

Therefore, by Lemma 4.2 and induction over t, every stored value in "down" mode is <= the exact U_t of the
relaxation, and the returned U_0 is <= v_n (Proposition 4.6).  (Assumptions: IEEE-754 binary64
round-to-nearest-even, no overflow -- all values are in [0, n+1] -- and subnormals handled by nextafter; lo = 0 makes m*lo = 0 and
dn(0) = -5e-324, which is harmless.)
"""
import numba as nb
import numpy as np
from numba.core import types
from numba.extending import intrinsic

from .encoding import rank_seq, unrank_seq, next_seq

NINF = -np.inf
PINF = np.inf


@intrinsic
def _f2i(typingctx, x):
    """reinterpret the bits of a float64 as int64 (LLVM bitcast; no conversion)."""
    sig = types.int64(types.float64)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], context.get_value_type(types.int64))
    return sig, codegen


@intrinsic
def _i2f(typingctx, x):
    sig = types.float64(types.int64)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], context.get_value_type(types.float64))
    return sig, codegen


@nb.njit(inline="always")
def dn(x):
    """= np.nextafter(x, -inf) for every float64 x (incl. +-0, subnormals, +-inf, nan), via the IEEE bit
    pattern: for finite x > 0 the predecessor has integer representation one smaller, for x < 0 one larger
    (sign-magnitude).  Cheaper than the library call inside the kernel.  Equality with np.nextafter is
    tested exhaustively on special values and 10^6 random values (test_dn_up_match_nextafter)."""
    if x > 0.0:
        if x < PINF:
            return _i2f(_f2i(x) - 1)
        return 1.7976931348623157e308
    if x < 0.0:
        if x > NINF:
            return _i2f(_f2i(x) + 1)
        return x
    if x == 0.0:
        return -5e-324
    return x          # nan


@nb.njit(inline="always")
def up(x):
    """= np.nextafter(x, +inf) for every float64 x."""
    if x < 0.0:
        if x > NINF:
            return _i2f(_f2i(x) - 1)
        return -1.7976931348623157e308
    if x > 0.0:
        if x < PINF:
            return _i2f(_f2i(x) + 1)
        return x
    if x == 0.0:
        return 5e-324
    return x          # nan


@nb.njit(cache=True)
def dn_up_vec(a):
    out = np.empty((2, a.shape[0]))
    for i in range(a.shape[0]):
        out[0, i] = dn(a[i])
        out[1, i] = up(a[i])
    return out


@nb.njit(inline="always")
def panel_near(lo, hi, h, r, m, w):
    clo = r + m * lo
    if w <= clo:
        return w * h
    chi = r + m * hi
    y = w if w < chi else chi
    d = y - clo
    return y * h - d * d / (2.0 * m)


@nb.njit(inline="always")
def panel_down(lo, hi, hd, r, m, w):
    clo_d = dn(r + dn(m * lo))
    if w <= clo_d:
        return dn(w * hd)
    clo_u = up(r + up(m * lo))
    chi_u = up(r + up(m * hi))
    y = w if w < chi_u else chi_u
    d1 = up(y - clo_d)
    d2 = up(clo_u - y)
    d = d1 if d1 > d2 else d2
    q = up(up(d * d) / (2.0 * m))
    g = dn(dn(y * hd) - q)
    c = dn(clo_d * hd)
    return g if g > c else c


@nb.njit(cache=True)
def closed_form_A(down, hf, cntK):
    """A[g] = sum_l h_l (1 + cnt[g, l])  (memory part of U_{n-1}); rounded down in 'down' mode."""
    nk, Gf = cntK.shape
    A = np.empty(nk)
    for g in range(nk):
        acc = 0.0
        for l in range(Gf):
            if down:
                acc = dn(acc + dn(hf[l] * (1.0 + cntK[g, l])))
            else:
                acc += hf[l] * (1.0 + cntK[g, l])
        A[g] = acc
    return A


@nb.njit(inline="always")
def closed_form_B(down, hE, S):
    acc = 0.0
    for b in range(S.shape[0]):
        if down:
            acc = dn(acc + dn(hE[b] * (1.0 + S[b])))
        else:
            acc += hE[b] * (1.0 + S[b])
    return acc


@nb.njit(cache=True, parallel=True)
def layer_kernel(down, store32, cf, n, t, k, f, E, hf, hE, caps,
                 ub, cnt, pref, base, bs,
                 cntn, prefn, basen,
                 bstart, nxt, nxtloc, cntK, A,
                 Unext, Ucur, chunk):
    """Compute layer t (states after t observations, next arrival is observation s = t+1 <= n-1).
    cf=True: layer t+1 = n-1 is not stored; its values come from the closed form A[K] + B(S)."""
    s = t + 1
    m = float(n - s)
    D = E.shape[0] - 1
    Gf = f.shape[0] - 1
    nseq = pref[0, ub[0] + 1]
    nchunks = (nseq + chunk - 1) // chunk
    for c in nb.prange(nchunks):
        S = np.empty(D, dtype=np.int64)
        Sn = np.empty(D, dtype=np.int64)
        succ = np.empty(D + 1, dtype=np.int64)   # succ[0]: fine arrival; succ[b]: coarse bin b (flat base)
        Bv = np.empty(D + 1)                     # closed-form B(S') when cf
        sid0 = c * chunk
        sid1 = min(nseq, sid0 + chunk)
        unrank_seq(sid0, cnt, S)
        for sid in range(sid0, sid1):
            v = S[0]
            j = v if v < k else k
            bsz = bs[j]
            gs = bstart[j]
            out0 = base[v] + (sid - pref[0, v]) * bsz
            # ---- successors
            for b in range(D + 1):
                for p in range(D):
                    if p >= b:            # b=0: fine arrival increments every S_p; bin b increments p >= b
                        x = S[p] + 1
                        Sn[p] = x if x < caps[p] else caps[p]
                    else:
                        Sn[p] = S[p]
                if cf:
                    Bv[b] = closed_form_B(down, hE, Sn)
                else:
                    vn = Sn[0]
                    jn = vn if vn < k else k
                    rn = rank_seq(Sn, prefn)
                    succ[b] = basen[vn] + (rn - prefn[0, vn]) * bs[jn]
            # ---- memory states of this block
            for lc in range(bsz):
                g = gs + lc
                acc = 0.0
                for l in range(Gf):
                    r = 1.0 + cntK[g, l]
                    if cf:
                        if down:
                            w = dn(A[nxt[g, l]] + Bv[0])
                        else:
                            w = A[nxt[g, l]] + Bv[0]
                    else:
                        w = np.float64(Unext[succ[0] + nxtloc[g, l]])
                    if down:
                        acc = dn(acc + panel_down(f[l], f[l + 1], hf[l], r, m, w))
                    else:
                        acc += panel_near(f[l], f[l + 1], hf[l], r, m, w)
                for b in range(1, D + 1):
                    r = 1.0 + S[b - 1]
                    if cf:
                        if down:
                            w = dn(A[g] + Bv[b])
                        else:
                            w = A[g] + Bv[b]
                    else:
                        w = np.float64(Unext[succ[b] + lc])
                    if down:
                        acc = dn(acc + panel_down(E[b - 1], E[b], hE[b - 1], r, m, w))
                    else:
                        acc += panel_near(E[b - 1], E[b], hE[b - 1], r, m, w)
                if store32:
                    v32 = np.float32(acc)
                    if down and np.float64(v32) > acc:
                        v32 = np.nextafter(v32, np.float32(-np.inf))
                    Ucur[out0 + lc] = v32
                else:
                    Ucur[out0 + lc] = acc
            next_seq(S, ub)


@nb.njit(cache=True)
def panel_vec(down, lo, hi, r, m, w):
    """vectorised single-cell panels (tests only); hd computed as in the solver."""
    out = np.empty(lo.shape[0])
    for i in range(lo.shape[0]):
        if down:
            hd = dn(hi[i] - lo[i])
            out[i] = panel_down(lo[i], hi[i], hd, r[i], m[i], w[i])
        else:
            out[i] = panel_near(lo[i], hi[i], hi[i] - lo[i], r[i], m[i], w[i])
    return out
