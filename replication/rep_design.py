"""Independent replication -- design (cell edges, caps), written from the written specification only.

Interpretation (see README.md, "Interpretation choices"):
  theta = th / n  (binary64)
  fine edges   f_l = theta * l / Gf,  l = 0..Gf, f_0 = 0, f_Gf = theta exactly          (bf = 0: uniform)
  coarse edges E_b = theta + (1-theta) * expm1(bc*b/D) / expm1(bc),  b = 0..D, E_0 = theta, E_D = 1
               (bc == 0: uniform)
  caps C_1..C_D apply to S_b = #{past obs <= E_{b-1}} (Lemma 3.5 of the paper); None = no caps (C_b = n)
Mathematical cells are the half-open intervals (lo, hi] between consecutive binary64 edges.
Cells are numbered 0..Gf-1 (fine) and Gf..Gf+D-1 (coarse bin b = 1..D).

The "pure k-best" relaxation (theta = 1, no coarse bins; Proposition 7.1 / Bruss-Ferguson W_n^(k)) is represented
as D = 1 with a zero-width coarse bin (1, 1] and cap C_1 = k: S_1 then equals min(k, #fine obs),
which is exactly the information the block structure needs, and the empty bin contributes 0.
"""
import math
from dataclasses import dataclass, field


@dataclass
class Design:
    n: int
    k: int
    fine: list           # Gf+1 edges, 0 .. theta
    coarse: list         # D+1 edges, theta .. 1
    caps: list           # D ints
    label: str = ""

    @property
    def Gf(self):
        return len(self.fine) - 1

    @property
    def D(self):
        return len(self.coarse) - 1

    def cells(self):
        """list of (lo, hi) for all Gf + D cells (fine first)."""
        out = [(self.fine[i], self.fine[i + 1]) for i in range(self.Gf)]
        out += [(self.coarse[i], self.coarse[i + 1]) for i in range(self.D)]
        return out

    def check(self):
        assert self.fine[0] == 0.0 and self.coarse[-1] == 1.0
        assert self.fine[-1] == self.coarse[0]
        for a, b in zip(self.fine, self.fine[1:]):
            assert a < b
        for a, b in zip(self.coarse, self.coarse[1:]):
            assert a <= b
        assert len(self.caps) == self.D
        assert all(c2 >= c1 for c1, c2 in zip(self.caps, self.caps[1:])), "caps must be nondecreasing"
        assert self.caps[0] >= self.k, "need C_1 >= k (block index = min(k, S_1))"
        return self


def hybrid(n, k, th, Gf, D, bc, caps=None, label=""):
    theta = th / n
    fine = [theta * l / Gf for l in range(Gf + 1)]
    fine[0] = 0.0
    fine[-1] = theta
    if bc == 0:
        coarse = [theta + (1.0 - theta) * b / D for b in range(D + 1)]
    else:
        coarse = [theta + (1.0 - theta) * math.expm1(bc * b / D) / math.expm1(bc) for b in range(D + 1)]
    coarse[0] = theta
    coarse[-1] = 1.0
    if caps is None:
        caps = [max(n, k)] * D
    return Design(n, k, fine, coarse, list(caps), label or f"hyb(n={n},k={k},th={th},Gf={Gf},D={D},bc={bc},caps={caps})").check()


def kbest(n, k, G, label=""):
    """theta = 1, no real coarse bin (zero-width bin (1,1], cap k)."""
    fine = [l / G for l in range(G + 1)]
    fine[-1] = 1.0
    return Design(n, k, fine, [1.0, 1.0], [max(k, 0)], label or f"kbest(n={n},k={k},G={G})").check()


def custom(n, k, fine, coarse, caps=None, label=""):
    D = len(coarse) - 1
    if caps is None:
        caps = [max(n, k)] * D
    return Design(n, k, list(fine), list(coarse), list(caps), label or "custom").check()
