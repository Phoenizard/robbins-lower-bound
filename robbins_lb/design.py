"""Design of the hybrid coarsening (Definition 3.3 of the paper) together with the horizon n.

A design fixes
  n     horizon;
  k     fine memory size (number of tracked smallest values <= theta), 0 <= k <= 4;
  f     fine edges   0 = f_0 < f_1 < ... < f_G = theta          (G = Gf fine cells (f_{l-1}, f_l]);
  E     coarse edges theta = E_0 < E_1 < ... < E_D = 1           (D coarse bins (E_{b-1}, E_b]);
  caps  optional caps C_1 <= ... <= C_D of the cumulative counts (see below).

Summary after t observations (Section 3.2, with cumulative counts):
  K    = sorted fine-cell indices of the k smallest observations <= theta, padded with EMPTY = G;
  S_b  = #{i <= t : X_i <= E_{b-1}},  b = 1..D   (S_1 = N_t = #obs <= theta; 0 <= S_1 <= ... <= S_D <= t).
  The count of the top bin, t - S_D, is never used by a relaxed rank, so it is not stored.
At fixed t this is a bijective re-encoding of the bin counts (N_t, N^1, ..., N^D):
S_b = N_t + N^1 + ... + N^{b-1}.

caps = None is the uncapped summary of Lemma 3.4.  caps = (C_1..C_D) stores min(S_b, C_b) instead of S_b
(the capped summary of Lemma 3.5; see the package docstring).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def geo_edges(lo: float, hi: float, cnt: int, beta: float) -> np.ndarray:
    """cnt cells on [lo, hi]; edges lo + (hi-lo)(e^{beta u}-1)/(e^beta-1), u = i/cnt (beta=0: uniform).
    beta > 0 concentrates cells near lo, beta < 0 near hi.  Endpoints are set exactly."""
    u = np.arange(cnt + 1) / cnt
    g = np.expm1(beta * u) / math.expm1(beta) if beta != 0 else u.copy()
    out = lo + (hi - lo) * g
    out[0] = lo
    out[-1] = hi
    return out


def dyadic(x: np.ndarray, bits: int) -> np.ndarray:
    """Round edges to the nearest multiple of 2^-bits (keeps endpoints; result strictly increasing is checked
    by Design).  Optional and not used by the certified designs: the rounding-down kernel does not need
    dyadic edges (it never relies on an exact branch selection, Lemma 4.4), but short dyadic edges make
    exact-rational replays cheaper."""
    s = float(2 ** bits)
    return np.round(np.asarray(x, dtype=np.float64) * s) / s


@dataclass(frozen=True)
class Design:
    n: int
    k: int
    f: tuple          # fine edges, floats, f[0]=0, f[-1]=theta
    E: tuple          # coarse edges, floats, E[0]=theta, E[-1]=1
    caps: tuple | None = None
    label: str = field(default="", compare=False)

    def __post_init__(self):
        f, E = np.asarray(self.f), np.asarray(self.E)
        if self.n < 1:
            raise ValueError("n >= 1")
        if not (0 <= self.k <= 4):
            raise ValueError("0 <= k <= 4")
        if len(f) < 2 or f[0] != 0.0 or not np.all(np.diff(f) > 0):
            raise ValueError("fine edges must start at 0 and increase strictly")
        if len(E) < 2 or E[-1] != 1.0 or not np.all(np.diff(E) > 0):
            raise ValueError("coarse edges must end at 1 and increase strictly (D >= 1)")
        if f[-1] != E[0]:
            raise ValueError("f[-1] must equal E[0] (= theta)")
        if self.caps is not None:
            c = np.asarray(self.caps)
            if len(c) != len(E) - 1:
                raise ValueError("need one cap per coarse bin")
            if np.any(np.diff(c) < 0):
                raise ValueError("caps must be nondecreasing")
            if c[0] < max(self.k, 1):
                raise ValueError("cap C_1 must be >= max(k, 1)")

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def hybrid(n, k, theta_n, Gf, D, bc, bf=0.0, caps=None, dyadic_bits=None, label=""):
        """Parametrisation of Section 5.1: theta = theta_n / n, Gf fine cells (geometric parameter bf,
        0 = uniform), D coarse bins with geometric parameter bc (the paper's beta).
        caps: None, an int (common cap), or a sequence of D ints."""
        theta = theta_n / n
        f = geo_edges(0.0, theta, Gf, bf)
        E = geo_edges(theta, 1.0, D, bc)
        if dyadic_bits is not None:
            f, E = dyadic(f, dyadic_bits), dyadic(E, dyadic_bits)
            E[0] = f[-1]
        if caps is not None and np.isscalar(caps):
            caps = (int(caps),) * D
        if caps is not None:
            caps = tuple(int(c) for c in caps)
        lab = label or f"n={n} k={k} th={theta_n} Gf={Gf} bf={bf} D={D} bc={bc} caps={caps}"
        return Design(n, k, tuple(float(x) for x in f), tuple(float(x) for x in E), caps, lab)

    # ------------------------------------------------------------------ derived sizes
    @property
    def Gf(self):
        return len(self.f) - 1

    @property
    def D(self):
        return len(self.E) - 1

    @property
    def theta(self):
        return self.f[-1]

    def block_sizes(self):
        """bs[j] = number of memory multisets with exactly j real entries (j = 0..k) = C(Gf+j-1, j)."""
        return [math.comb(self.Gf + j - 1, j) for j in range(self.k + 1)]

    def ub(self, t):
        """upper bounds of S_1..S_D in layer t (after t observations)."""
        if self.caps is None:
            return [t] * self.D
        return [min(t, c) for c in self.caps]

    def layer_states(self, t):
        """number of stored states (K, S) in layer t, counting only reachable memory blocks
        (#real entries of K = min(k, S_1))."""
        from .encoding import count_table
        cnt = count_table(self.ub(t))       # cnt[p][v] = #valid suffixes with S_p = v
        bs = self.block_sizes()
        # number of sequences with S_1 = v is cnt[0][v]
        return int(sum(int(cnt[0][v]) * bs[min(self.k, v)] for v in range(len(cnt[0]))))

    def max_layer_states(self):
        return max(self.layer_states(t) for t in range(self.n))

    def peak_bytes(self, storage="f64"):
        """two live layers (t, t+1); the last layer t=n-1 is not stored (computed in closed form)."""
        b = 8 if storage == "f64" else 4
        sz = [self.layer_states(t) for t in range(self.n)]
        if self.n == 1:
            return 0
        peak = max(sz[t] + (sz[t + 1] if t + 1 <= self.n - 2 else 0) for t in range(self.n - 1))
        return peak * b

    def panels(self):
        return sum(self.layer_states(t) for t in range(self.n)) * (self.Gf + self.D)

    def to_dict(self):
        return dict(n=self.n, k=self.k, f=list(self.f), E=list(self.E),
                    caps=None if self.caps is None else list(self.caps), label=self.label)

    @staticmethod
    def from_dict(d):
        caps = d.get("caps")
        return Design(int(d["n"]), int(d["k"]), tuple(d["f"]), tuple(d["E"]),
                      None if caps is None else tuple(caps), d.get("label", ""))
