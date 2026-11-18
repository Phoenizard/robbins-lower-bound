"""Information-relaxation (pathwise) dual lower bound and primal upper bound for Robbins' problem.

For any approximate value function, set for s = 1..n
    Vt_s = min(c_s, What_s(state_s))  (s < n),   Vt_n = c_n,
    c_s  = r_s + (n - s) X_s = E[R_s | F_s],
    M_s  = sum_{u<=s} ( Vt_u - E[Vt_u | F_{u-1}] ).
M is an F-martingale with M_0 = 0 provided the conditional expectations are exact; then
    v_n >= E[ min_{1<=j<=n} (c_j - M_j) ]      (weak duality; Rogers / Haugh-Kogan / BSS).
E[Vt_u | F_{u-1}] = int_0^1 min(c_u(x), What_u(state(x))) dx is computed EXACTLY (up to rounding):
What is bilinear on the grid, so along every line in x it is piecewise linear with kinks at grid
nodes; c_u(x) is piecewise linear with jumps at the past values.  Each panel integral of
min(linear, linear) is evaluated in closed form.

The primal policy stops at the first s with c_s <= What_s(state_s) (or at n).  Its MC mean of
c_tau is an unbiased estimate of its expected rank, an upper bound on v_n.

Memory models supported: k = 2 (state (a1,a2), W of shape (n,G,G)) and k = 3 (state (a1,a2,a3),
W of shape (n,G,G,G)).
"""
import math
import numpy as np
import numba as nb


# ----------------------------------------------------------------------------------------------
@nb.njit(cache=True, inline="always")
def _panel(h, c0, c1, w0, w1):
    d0 = c0 - w0
    d1 = c1 - w1
    base = 0.5 * h * (c0 + c1)
    if d0 >= 0.0 and d1 >= 0.0:
        return base - 0.5 * h * (d0 + d1)
    if d0 <= 0.0 and d1 <= 0.0:
        return base
    pos = d0 if d0 > d1 else d1
    return base - 0.5 * h * pos * pos / (abs(d0) + abs(d1))


@nb.njit(cache=True, inline="always")
def _locate(a, G, beta, em1b):
    """index i with g_i <= a <= g_{i+1}, and the fractional weight."""
    if beta == 0.0:
        u = a
    else:
        u = math.log1p(a * em1b) / beta
    f = u * (G - 1)
    i = int(math.floor(f))
    if i < 0:
        i = 0
    if i > G - 2:
        i = G - 2
    return i


@nb.njit(cache=True)
def _interp2(W, g, a, b, beta, em1b):
    G = g.shape[0]
    i = _locate(a, G, beta, em1b)
    l = _locate(b, G, beta, em1b)
    # guard against rounding in the inverse map
    while i > 0 and a < g[i]:
        i -= 1
    while i < G - 2 and a > g[i + 1]:
        i += 1
    while l > 0 and b < g[l]:
        l -= 1
    while l < G - 2 and b > g[l + 1]:
        l += 1
    la = (a - g[i]) / (g[i + 1] - g[i])
    mu = (b - g[l]) / (g[l + 1] - g[l])
    return ((1 - la) * (1 - mu) * W[i, l] + la * (1 - mu) * W[i + 1, l]
            + (1 - la) * mu * W[i, l + 1] + la * mu * W[i + 1, l + 1])


@nb.njit(cache=True)
def _interp3(W, g, a, b, c, beta, em1b):
    G = g.shape[0]
    i = _locate(a, G, beta, em1b)
    l = _locate(b, G, beta, em1b)
    k = _locate(c, G, beta, em1b)
    while i > 0 and a < g[i]:
        i -= 1
    while i < G - 2 and a > g[i + 1]:
        i += 1
    while l > 0 and b < g[l]:
        l -= 1
    while l < G - 2 and b > g[l + 1]:
        l += 1
    while k > 0 and c < g[k]:
        k -= 1
    while k < G - 2 and c > g[k + 1]:
        k += 1
    la = (a - g[i]) / (g[i + 1] - g[i])
    mu = (b - g[l]) / (g[l + 1] - g[l])
    nu = (c - g[k]) / (g[k + 1] - g[k])
    v = 0.0
    for di in range(2):
        wi = la if di == 1 else 1 - la
        for dl in range(2):
            wl = mu if dl == 1 else 1 - mu
            for dk in range(2):
                wk = nu if dk == 1 else 1 - nu
                v += wi * wl * wk * W[i + di, l + dl, k + dk]
    return v


# ----------------------------------------------------------------------------------------------
# Generic line integral: int_lo^hi min(r + m x, What(state(x))) dx, where state(x) is obtained by
# inserting x at a fixed position among the remembered values; What is evaluated with the
# appropriate interpolant.  `which` encodes the insert position.
@nb.njit(cache=True)
def _line_int2(W, g, beta, em1b, lo, hi, r, m, which, a1, a2):
    """k=2: which=0 -> state (x, a1); which=1 -> state (a1, x)."""
    if hi <= lo:
        return 0.0
    G = g.shape[0]
    j = _locate(lo, G, beta, em1b)
    while j < G - 2 and g[j + 1] <= lo:
        j += 1
    while j > 0 and g[j] > lo:
        j -= 1
    x0 = lo
    if which == 0:
        w0 = _interp2(W, g, x0, a1, beta, em1b)
    else:
        w0 = _interp2(W, g, a1, x0, beta, em1b)
    tot = 0.0
    while True:
        x1 = g[j + 1] if g[j + 1] < hi else hi
        if which == 0:
            w1 = _interp2(W, g, x1, a1, beta, em1b)
        else:
            w1 = _interp2(W, g, a1, x1, beta, em1b)
        tot += _panel(x1 - x0, r + m * x0, r + m * x1, w0, w1)
        if x1 >= hi:
            break
        x0 = x1
        w0 = w1
        j += 1
        if j > G - 2:
            break
    return tot


@nb.njit(cache=True)
def _line_int3(W, g, beta, em1b, lo, hi, r, m, which, a1, a2, a3):
    """k=3: which=0 -> (x,a1,a2); 1 -> (a1,x,a2); 2 -> (a1,a2,x)."""
    if hi <= lo:
        return 0.0
    G = g.shape[0]
    j = _locate(lo, G, beta, em1b)
    while j < G - 2 and g[j + 1] <= lo:
        j += 1
    while j > 0 and g[j] > lo:
        j -= 1
    x0 = lo
    if which == 0:
        w0 = _interp3(W, g, x0, a1, a2, beta, em1b)
    elif which == 1:
        w0 = _interp3(W, g, a1, x0, a2, beta, em1b)
    else:
        w0 = _interp3(W, g, a1, a2, x0, beta, em1b)
    tot = 0.0
    while True:
        x1 = g[j + 1] if g[j + 1] < hi else hi
        if which == 0:
            w1 = _interp3(W, g, x1, a1, a2, beta, em1b)
        elif which == 1:
            w1 = _interp3(W, g, a1, x1, a2, beta, em1b)
        else:
            w1 = _interp3(W, g, a1, a2, x1, beta, em1b)
        tot += _panel(x1 - x0, r + m * x0, r + m * x1, w0, w1)
        if x1 >= hi:
            break
        x0 = x1
        w0 = w1
        j += 1
        if j > G - 2:
            break
    return tot


@nb.njit(cache=True)
def _tail_int(sorted_past, npast, start_idx, lo, r0, m, wconst):
    """int_lo^1 min(r(x) + m x, wconst) dx, r(x) = r0 + #{past values in (lo, x)}.

    sorted_past[start_idx:npast] are the past values > lo, ascending.  Exact."""
    if lo >= 1.0:
        return 0.0
    # int min(c, w) = w * (1 - lo) - int (w - c)^+ ; (w - c)^+ = 0 once r + m x >= w.
    tot = wconst * (1.0 - lo)
    x0 = lo
    r = r0
    idx = start_idx
    while True:
        if r + m * x0 >= wconst:
            break
        x1 = sorted_past[idx] if idx < npast else 1.0
        # on (x0, x1): c = r + m x linear increasing; (w - c)^+
        if m > 0.0:
            xs = (wconst - r) / m  # where c hits w
        else:
            xs = 2.0
        xe = x1 if x1 < xs else xs
        if xe > x0:
            # int_x0^xe (w - r - m x) dx
            tot -= (wconst - r) * (xe - x0) - 0.5 * m * (xe * xe - x0 * x0)
        if idx >= npast or xs <= x1:
            break
        x0 = x1
        r += 1.0
        idx += 1
    return tot


@nb.njit(cache=True)
def _insert_sorted(arr, npast, x):
    """insert x into arr[0:npast] (ascending); return position."""
    p = npast
    while p > 0 and arr[p - 1] > x:
        arr[p] = arr[p - 1]
        p -= 1
    arr[p] = x
    return p


# ----------------------------------------------------------------------------------------------
@nb.njit(cache=True, parallel=True)
def simulate_k2(X, W, g, beta, Cp, Mp, rec):
    """X: (P, n) uniforms. W: (n, G, G). If rec, store c_s and M_s in Cp, Mp (P, n).
    Returns (dual, primal, tau, Mn) per path."""
    P, n = X.shape
    em1b = math.expm1(beta) if beta != 0.0 else 1.0
    dual = np.empty(P)
    primal = np.empty(P)
    tau = np.empty(P, dtype=np.int64)
    Mn = np.empty(P)
    for p in nb.prange(P):
        past = np.empty(n + 1)
        npast = 0
        a1 = 1.0
        a2 = 1.0
        M = 0.0
        best = 1e300
        stopped = False
        pv = 0.0
        tp = n
        for s in range(1, n + 1):
            m = float(n - s)
            # ---- E[Vt_s | F_{s-1}] -----------------------------------------------------------
            if s == n:
                # Vt = c exactly: int (r(x) + 0) dx = 1 + sum_i (1 - X_i)
                ev = 1.0
                for q in range(npast):
                    ev += 1.0 - past[q]
            else:
                Ws = W[s]
                ev = _line_int2(Ws, g, beta, em1b, 0.0, a1, 1.0, m, 0, a1, a2)
                ev += _line_int2(Ws, g, beta, em1b, a1, a2, 2.0, m, 1, a1, a2)
                # tail: past values > a2 start at index 2 (when npast >= 2)
                st = 2 if npast >= 2 else npast
                wc = _interp2(Ws, g, a1, a2, beta, em1b)
                ev += _tail_int(past, npast, st, a2, 3.0, m, wc)
            # ---- observe X_s ------------------------------------------------------------------
            x = X[p, s - 1]
            pos = _insert_sorted(past, npast, x)
            npast += 1
            r = float(pos + 1)
            c = r + m * x
            if x < a1:
                a2 = a1
                a1 = x
            elif x < a2:
                a2 = x
            if s == n:
                vt = c
                wcur = 1e300
            else:
                wcur = _interp2(W[s], g, a1, a2, beta, em1b)
                vt = c if c < wcur else wcur
            M += vt - ev
            if rec:
                Cp[p, s - 1] = c
                Mp[p, s - 1] = M
            d = c - M
            if d < best:
                best = d
            if (not stopped) and c <= wcur:
                stopped = True
                pv = c
                tp = s
        dual[p] = best
        primal[p] = pv
        tau[p] = tp
        Mn[p] = M
    return dual, primal, tau, Mn


@nb.njit(cache=True, parallel=True)
def simulate_k3(X, W, g, beta, Cp, Mp, rec):
    """X: (P, n) uniforms. W: (n, G, G, G)."""
    P, n = X.shape
    em1b = math.expm1(beta) if beta != 0.0 else 1.0
    dual = np.empty(P)
    primal = np.empty(P)
    tau = np.empty(P, dtype=np.int64)
    Mn = np.empty(P)
    for p in nb.prange(P):
        past = np.empty(n + 1)
        npast = 0
        a1 = 1.0
        a2 = 1.0
        a3 = 1.0
        M = 0.0
        best = 1e300
        stopped = False
        pv = 0.0
        tp = n
        for s in range(1, n + 1):
            m = float(n - s)
            if s == n:
                ev = 1.0
                for q in range(npast):
                    ev += 1.0 - past[q]
            else:
                Ws = W[s]
                ev = _line_int3(Ws, g, beta, em1b, 0.0, a1, 1.0, m, 0, a1, a2, a3)
                ev += _line_int3(Ws, g, beta, em1b, a1, a2, 2.0, m, 1, a1, a2, a3)
                ev += _line_int3(Ws, g, beta, em1b, a2, a3, 3.0, m, 2, a1, a2, a3)
                st = 3 if npast >= 3 else npast
                wc = _interp3(Ws, g, a1, a2, a3, beta, em1b)
                ev += _tail_int(past, npast, st, a3, 4.0, m, wc)
            x = X[p, s - 1]
            pos = _insert_sorted(past, npast, x)
            npast += 1
            r = float(pos + 1)
            c = r + m * x
            if x < a1:
                a3 = a2
                a2 = a1
                a1 = x
            elif x < a2:
                a3 = a2
                a2 = x
            elif x < a3:
                a3 = x
            if s == n:
                vt = c
                wcur = 1e300
            else:
                wcur = _interp3(W[s], g, a1, a2, a3, beta, em1b)
                vt = c if c < wcur else wcur
            M += vt - ev
            if rec:
                Cp[p, s - 1] = c
                Mp[p, s - 1] = M
            d = c - M
            if d < best:
                best = d
            if (not stopped) and c <= wcur:
                stopped = True
                pv = c
                tp = s
        dual[p] = best
        primal[p] = pv
        tau[p] = tp
        Mn[p] = M
    return dual, primal, tau, Mn


# ----------------------------------------------------------------------------------------------
# Brute-force reference for E[Vt_s | F_{s-1}] (used only in tests): dense midpoint rule.
def brute_ev_k2(W, g, beta, n, s, past_vals, npts=2_000_000):
    em1b = math.expm1(beta) if beta != 0.0 else 1.0
    past = np.sort(np.asarray(past_vals, dtype=float))
    a1 = past[0] if len(past) >= 1 else 1.0
    a2 = past[1] if len(past) >= 2 else 1.0
    xs = (np.arange(npts) + 0.5) / npts
    r = 1.0 + np.searchsorted(past, xs)
    c = r + (n - s) * xs
    if s == n:
        return float(np.mean(c))
    na1 = np.where(xs < a1, xs, a1)
    na2 = np.where(xs < a1, a1, np.where(xs < a2, xs, a2))
    w = np.array([_interp2(W[s], g, u, v, beta, em1b) for u, v in zip(na1, na2)])
    return float(np.mean(np.minimum(c, w)))


def run_sim(k, X, W, g, beta, record=False):
    P, n = X.shape
    if record:
        Cp = np.empty((P, n)); Mp = np.empty((P, n))
    else:
        Cp = np.empty((1, 1)); Mp = np.empty((1, 1))
    f = simulate_k2 if k == 2 else simulate_k3
    d, pr, tau, Mn = f(X, W, g, beta, Cp, Mp, record)
    if record:
        return d, pr, tau, Mn, Cp, Mp
    return d, pr, tau, Mn
