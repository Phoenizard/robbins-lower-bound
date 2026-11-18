"""Design specifications (Section 5.1 of the paper) and the two certified designs.

A *spec* is a dict {n, k, th, Gf, bf, D, bc, caps} with
  th    theta * n (so theta = th / n);
  Gf    number of fine cells G, with geometric parameter bf (0 = uniform, the only value used in the paper);
  D     number of coarse bins, with geometric parameter bc (the paper's beta);
  caps  None (uncapped, Lemma 3.4), an int (common cap), a list of D ints, or the rule [a, c] / [a, z, c]
        (floats) meaning C_b = ceil(a*n*e + z*sqrt(n*e*(1-e)) + c) with e = E_{b-1}, then made nondecreasing
        and at least max(k, 1) (capped summary, Lemma 3.5).

build(spec) is the function that produced the designs of the two original certificate records
(certificates/original/); it is copied without change from the research code.
"""
import math

import numpy as np

from .design import Design


def build(spec):
    caps = spec.get("caps")
    if isinstance(caps, list) and len(caps) in (2, 3) and isinstance(caps[0], float):
        d0 = Design.hybrid(spec["n"], spec["k"], spec["th"], spec["Gf"], spec["D"], spec["bc"], spec.get("bf", 0.0))
        a, z, c = (caps[0], 0.0, caps[1]) if len(caps) == 2 else caps
        nn = spec["n"]
        cc = [max(max(spec["k"], 1), math.ceil(a * nn * e + z * math.sqrt(nn * e * (1 - e)) + c))
              for e in d0.E[:-1]]
        caps = [int(x) for x in np.maximum.accumulate(cc)]
    return Design.hybrid(spec["n"], spec["k"], spec["th"], spec["Gf"], spec["D"], spec["bc"], spec.get("bf", 0.0),
                         caps=caps)


# Main design of Theorem 6.1 (the paper's D_100): caps (8, 9, 10, 14, 23, 44).
SPEC_D100 = {"n": 100, "k": 3, "th": 7, "Gf": 20, "bf": 0.0, "D": 6, "bc": 5.0, "caps": [0.9, 0.0, 1.0]}
# Fallback design of Corollary 6.2 (the paper's D_80): no caps.
SPEC_D80 = {"n": 80, "k": 3, "th": 7, "Gf": 12, "bf": 0.0, "D": 4, "bc": 3.0, "caps": None}
# Small capped design used by scripts/quick_check.py (not part of any theorem).
SPEC_N30 = {"n": 30, "k": 3, "th": 7, "Gf": 24, "bf": 0.0, "D": 5, "bc": 3.0, "caps": [1.0, 2.0, 2.0]}

D100 = build(SPEC_D100)
D80 = build(SPEC_D80)

# Results of the original certified runs (certificates/original/), as exact binary fractions.
L0_D100 = "4397361/2097152"     # = 2.096825122833251953125 = 0x1.0c64c40000000p+1
L0_D80 = "8603877/4194304"      # = 2.0513241291046142578125 = 0x1.0691ca0000000p+1
