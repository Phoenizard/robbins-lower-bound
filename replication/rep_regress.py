"""Independent replication -- regression against earlier floating-point values of four uncapped designs
(4 decimals, computed by the main implementation; also checked in tests/test_relax.py)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rep_env import setup_env
setup_env(4)
import rep_design as RD
import rep_solver as RS

CASES = [  # (n, k, th, Gf, D, bc, reference value)
    (10, 2, 4, 30, 8, 1.0, 1.7574),
    (20, 2, 6, 20, 6, 1.5, 1.8999),
    (30, 1, 5, 20, 5, 1.5, 1.9352),
    (30, 2, 6, 20, 5, 1.5, 1.9622),
]
rows = []
for n, k, th, Gf, D, bc, ref in CASES:
    d = RD.hybrid(n, k, th, Gf, D, bc, None)
    r = RS.solve(d, "nearest", "f64")
    rows.append(dict(design=d.label, U0=r["U0"], ref=ref, diff=r["U0"] - ref, seconds=r["seconds"]))
    print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(HERE, "results", "t3_regression.json"), "w"), indent=1)
