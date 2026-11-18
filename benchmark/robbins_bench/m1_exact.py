"""Independent high-precision solver for the m = 1 column of Bruss-Ferguson Table 3.

For m = 1 the state is the single smallest value y seen so far and recursion (4.5) reads
    V_0 == 1,    V_{k+1}(y) = int_0^y min(k x, V_k(x)) dx + (1 - y) V_k(y),      W_n^(1) = 1 + V_n(1).
Each V_k is a piecewise polynomial; V_k is nonincreasing, so k x = V_k(x) has at most one root s_k in [0,1].
We carry the pieces exactly as polynomials with mpmath coefficients (dps digits) and locate s_k by bisection
on the relevant piece.  The result is accurate to roughly dps - 10 digits (no discretisation at all); it is a
fully independent cross-check of the grid DP in truncated_dp.py.
"""
import mpmath as mp


def _peval(c, x):
    r = mp.mpf(0)
    for a in reversed(c):
        r = r * x + a
    return r


def _pmul(a, b):
    out = [mp.mpf(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def _pint(c):
    return [mp.mpf(0)] + [c[i] / (i + 1) for i in range(len(c))]


def W1_values(nmax, dps=60):
    """Return [W_1^(1), ..., W_nmax^(1)] as mpf."""
    with mp.workdps(dps):
        # pieces: list of (lo, hi, coeffs) covering [0,1]
        V = [(mp.mpf(0), mp.mpf(1), [mp.mpf(1)])]
        out = []
        for k in range(nmax):
            # crossing s_k of k x = V_k(x)
            if k == 0:
                s = mp.mpf(1)    # min(0, V_0) = 0 everywhere: integrand 0 = k x on all of [0, 1]
            else:
                s = None
                for lo, hi, c in V:
                    g_lo = k * lo - _peval(c, lo)
                    g_hi = k * hi - _peval(c, hi)
                    if g_lo <= 0 <= g_hi:
                        a, b = lo, hi
                        for _ in range(4 * dps):
                            mid = (a + b) / 2
                            if k * mid - _peval(c, mid) <= 0:
                                a = mid
                            else:
                                b = mid
                        s = (a + b) / 2
                        break
                if s is None:
                    s = mp.mpf(1)
            # new breakpoints
            bps = sorted(set([lo for lo, _, _ in V] + [mp.mpf(1)] + [s]))
            newV = []
            acc = k * s * s / 2      # int_0^s k x dx
            for i in range(len(bps) - 1):
                lo, hi = bps[i], bps[i + 1]
                mid = (lo + hi) / 2
                c = next(cc for (l, h, cc) in V if l <= mid <= h)
                one_minus_y_times = _pmul([mp.mpf(1), mp.mpf(-1)], c)
                if hi <= s:
                    I = [mp.mpf(0), mp.mpf(0), mp.mpf(k) / 2]
                    newV.append((lo, hi, _padd(I, one_minus_y_times)))
                else:
                    Q = _pint(c)
                    # int_s^y V_k = acc_piece + Q(y) - Q(lo)
                    const = acc - _peval(Q, lo)
                    I = _padd([const], Q)
                    newV.append((lo, hi, _padd(I, one_minus_y_times)))
                    acc = acc + _peval(Q, hi) - _peval(Q, lo)
            V = newV
            out.append(1 + _peval(V[-1][2], mp.mpf(1)))
        return out
