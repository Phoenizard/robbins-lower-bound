"""Rigorous two-sided enclosures of Bruss-Ferguson's truncated values W_n^(m).

Definitions (Bruss & Ferguson 1993, J. Appl. Probab. 30, Sec. 4; UCLA preprint with the 2005 bold-face
correction).  X_1..X_n iid U(0,1).  Loss for stopping at k (eq. 4.4):

    Y_k = 1 + min{ m, sum_{i<k} I(X_i < X_k) } + (n - k) X_k ,      W_n^(m) = inf_N E Y_N .

Value function (eq. 4.5), "minus 1", with k observations to go and state = the m smallest values seen so
far 0 < y_1 <= ... <= y_m <= 1 (unobserved order statistics replaced by 1):

    V_{k+1}(y) = sum_{j=0}^{m-1} int_{y_j}^{y_{j+1}} min{ k x + j , V_k(y_1..y_j, x, y_{j+1}..y_{m-1}) } dx
                 + (1 - y_m) V_k(y) ,        y_0 := 0,   V_0 == m,
    W_n^(m) = 1 + V_n(1, ..., 1).

Rigorous discretisation (this module).
  * Monotonicity: V_k is nonincreasing in every coordinate y_i (pathwise coupling: raising a stored value
    can only lower every future truncated count, and the insertion map is monotone).
  * Fix a grid 0 = t_0 < ... < t_G = 1 and restrict states to grid tuples.  For x in the cell
    (t_c, t_{c+1}) the state after inserting x is bracketed by inserting t_{c+1} (gives a smaller V) or
    t_c (gives a larger V).  Replacing V_k by these values inside the integral gives
        L_{k+1}(y) <= V_{k+1}(y) <= U_{k+1}(y)   for all grid tuples y,
    by induction (min and integral are monotone).  The integrals int_cell min(kx + j, C) dx are evaluated in
    closed form.
  * Floating point: every operation is rounded outward (np.nextafter on the round-to-nearest result,
    which is a valid one-sided bound for a correctly rounded +,-,*,/).  The grid is dyadic so the
    affine data k*a + j, b - a, a + b are exact.  The lower pass thus returns a number <= L (<= V), the
    upper pass a number >= U (>= V): together a rigorous enclosure (one-sided interval arithmetic).

State storage: nondecreasing m-tuples z_0 <= ... <= z_{m-1} of grid indices in {0..G}, ranked in colex order
    rank(z) = sum_i C(z_i + i, i + 1),     N = C(G + m, m).
"""
import math
import time

import numba as nb
import numpy as np


# ----------------------------------------------------------------------------------------------
# directed rounding primitives
# ----------------------------------------------------------------------------------------------
@nb.njit(inline="always")
def rd(x):
    """A float <= the exact value whose round-to-nearest is x (valid for one correctly rounded op)."""
    return np.nextafter(x, -np.inf)


@nb.njit(inline="always")
def ru(x):
    return np.nextafter(x, np.inf)


@nb.njit
def f_lo(C, a, b, k, j):
    """Lower bound of int_a^b min(k x + j, C) dx (a, b dyadic, k, j small ints)."""
    w = b - a                     # exact
    if k == 0:
        c = C if C < j else float(j)
        return rd(c * w)
    lo_line = k * a + j           # exact
    hi_line = k * b + j           # exact
    if C <= lo_line:
        return rd(C * w)
    if C >= hi_line:
        s = rd(w * (a + b))       # a + b exact
        s2 = rd(s * k)
        return rd(w * j + s2 * 0.5)   # w*j exact, *0.5 exact
    # 0 < D = C - lo_line < k w :  value = C w - D^2 / (2k)
    D = ru(C - lo_line)
    q = ru(D * D)
    q2 = ru(q / (2.0 * k))
    return rd(rd(C * w) - q2)


@nb.njit
def f_hi(C, a, b, k, j):
    """Upper bound of int_a^b min(k x + j, C) dx."""
    w = b - a
    if k == 0:
        c = C if C < j else float(j)
        return ru(c * w)
    lo_line = k * a + j
    hi_line = k * b + j
    if C <= lo_line:
        return ru(C * w)
    if C >= hi_line:
        s = ru(w * (a + b))
        s2 = ru(s * k)
        return ru(w * j + s2 * 0.5)
    D = rd(C - lo_line)
    if D < 0.0:
        D = 0.0
    q = rd(D * D)
    q2 = rd(q / (2.0 * k))
    return ru(ru(C * w) - q2)


@nb.njit
def f_nearest(C, a, b, k, j):
    """Plain round-to-nearest evaluation (used only by the non-rigorous midpoint variant/tests)."""
    w = b - a
    if k == 0:
        return min(C, float(j)) * w
    lo_line = k * a + j
    hi_line = k * b + j
    if C <= lo_line:
        return C * w
    if C >= hi_line:
        return w * (j + k * (a + b) * 0.5)
    D = C - lo_line
    return C * w - D * D / (2.0 * k)


# ----------------------------------------------------------------------------------------------
# tuple ranking
# ----------------------------------------------------------------------------------------------
def binom_table(nmax, rmax):
    B = np.zeros((nmax + 1, rmax + 2), dtype=np.int64)
    for a in range(nmax + 1):
        for r in range(rmax + 2):
            B[a, r] = math.comb(a, r)
    return B


@nb.njit(inline="always")
def rank_of(z, m, B):
    r = 0
    for i in range(m):
        r += B[z[i] + i, i + 1]
    return r


@nb.njit(inline="always")
def advance(z, m, G):
    """colex successor of a nondecreasing tuple; returns False after the last one."""
    i = 0
    while i < m:
        bound = z[i + 1] if i + 1 < m else G
        if z[i] < bound:
            z[i] += 1
            for q in range(i):
                z[q] = 0
            return True
        i += 1
    return False


# ----------------------------------------------------------------------------------------------
# one backward step  V_k -> V_{k+1}
# ----------------------------------------------------------------------------------------------
@nb.njit
def dp_step(Vk, k, m, G, t, B, lower):
    N = Vk.shape[0]
    out = np.empty(N)
    P = np.empty(N)
    z = np.zeros(m, dtype=np.int64)
    d = np.zeros(m, dtype=np.int64)

    # tail term (1 - y_m) V_k(y): arrivals above the m-th smallest never stop (V_k <= m <= stop cost)
    for q in range(m):
        z[q] = 0
    for idx in range(N):
        tail = 1.0 - t[z[m - 1]]          # exact
        if lower:
            out[idx] = rd(tail * Vk[idx])
        else:
            out[idx] = ru(tail * Vk[idx])
        advance(z, m, G)

    for j in range(m):
        # ---- pass 1: prefix sums along coordinate j ----
        for q in range(m):
            z[q] = 0
        for idx in range(N):
            low = z[j - 1] if j > 0 else 0
            zj = z[j]
            if lower:
                # P(z) = sum_{p=low+1}^{z_j} f_lo(cell (t_{p-1}, t_p), V_k(z with z_j = p))
                if zj > low:
                    A = f_lo(Vk[idx], t[zj - 1], t[zj], k, j)
                    if zj - 1 > low:
                        prev = idx - B[zj + j - 1, j]
                        P[idx] = rd(P[prev] + A)
                    else:
                        P[idx] = A
                else:
                    P[idx] = 0.0
            else:
                # Q(z) = sum_{p=low}^{z_j} f_hi(cell (t_p, t_{p+1}), V_k(z with z_j = p))
                if zj < G:
                    A = f_hi(Vk[idx], t[zj], t[zj + 1], k, j)
                    if zj > low:
                        prev = idx - B[zj + j - 1, j]
                        P[idx] = ru(P[prev] + A)
                    else:
                        P[idx] = A
                else:
                    P[idx] = 0.0
            advance(z, m, G)
        # ---- pass 2: add the j-th integral to every state ----
        for q in range(m):
            z[q] = 0
        for idx in range(N):
            low = z[j - 1] if j > 0 else 0
            if z[j] > low:
                for q in range(j):
                    d[q] = z[q]
                for q in range(j + 1, m):
                    d[q] = z[q - 1]
                if lower:
                    d[j] = z[j]
                    out[idx] = rd(out[idx] + P[rank_of(d, m, B)])
                else:
                    d[j] = z[j] - 1
                    out[idx] = ru(out[idx] + P[rank_of(d, m, B)])
            advance(z, m, G)
    return out


def n_states(G, m):
    return math.comb(G + m, m)


def enclose_W(m, t, nmax, lower, verbose=False):
    """Return list of rigorous one-sided bounds for W_n^(m), n = 1..nmax (index 0 -> n = 1).

    lower=True  -> each entry <= W_n^(m);  lower=False -> each entry >= W_n^(m).
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    G = len(t) - 1
    B = binom_table(G + m + 1, m)
    N = n_states(G, m)
    V = np.full(N, float(m))
    res = []
    t0 = time.time()
    for k in range(nmax):            # V_k -> V_{k+1}; the arriving observation has k still to come
        V = dp_step(V, k, m, G, t, B, lower)
        top = V[N - 1]               # state (1, ..., 1)
        W = float(np.nextafter(1.0 + top, -np.inf if lower else np.inf))
        res.append(W)
        if verbose:
            print(f"  m={m} G={G} {'L' if lower else 'U'} n={k + 1}: {W:.10f}  ({time.time() - t0:.1f}s)",
                  flush=True)
    return res


def value_tables(m, t, nmax, lower):
    """Return [V_0, V_1, ..., V_nmax] (arrays over grid tuples) of the lower or upper discrete recursion."""
    t = np.ascontiguousarray(t, dtype=np.float64)
    G = len(t) - 1
    B = binom_table(G + m + 1, m)
    V = np.full(n_states(G, m), float(m))
    out = [V]
    for k in range(nmax):
        V = dp_step(V, k, m, G, t, B, lower)
        out.append(V)
    return out


def enclose_W_both(m, t, nmax, verbose=False):
    lo = enclose_W(m, t, nmax, True, verbose)
    hi = enclose_W(m, t, nmax, False, verbose)
    return lo, hi
