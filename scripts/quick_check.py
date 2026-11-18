"""Quick regression check of the certified pipeline on a small design (about 5-12 minutes at 4 threads).

Usage:  python scripts/quick_check.py [--threads 4] [--also-nearest-f32]

Runs the rounded-down backward pass ("down" mode, binary64 storage) on the capped n = 30 design
SPEC_N30 of robbins_lb/designs.py,
    {"n":30,"k":3,"th":7,"Gf":24,"bf":0.0,"D":5,"bc":3.0,"caps":[1.0,2.0,2.0]},  caps (14, 15, 17, 21, 27),
and checks that the result is bit-identical to the value produced by the research code from which this
repository was derived:
    down,    f64 storage:  1.9727237588595965   (0x1.f9046c9c581abp+0)
    nearest, f32 storage:  1.9727237224578857   (0x1.f9046c0000000p+0; only with --also-nearest-f32)
The result is independent of the number of threads (parallelism is over disjoint output summaries only).
The value is a rigorous lower bound of v_30 (by the same argument as Theorem 6.1); it is not used in the paper.

Resources: the two live layers need 2.52 GiB with binary64 storage (1.26 GiB with binary32); 1.31e9 summaries.

If NUMBA_CACHE_DIR is not set, a new empty temporary directory is used, so that all kernels are compiled
from the current source.  Exit status 0 iff every check matches bit for bit.
"""
import argparse
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

EXPECTED = {
    ("down", "f64"): "0x1.f9046c9c581abp+0",       # 1.9727237588595965
    ("nearest", "f32"): "0x1.f9046c0000000p+0",    # 1.9727237224578857
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--also-nearest-f32", action="store_true")
    a = ap.parse_args()

    if not os.environ.get("NUMBA_CACHE_DIR"):
        os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="numba_cache_quick_")
    print("NUMBA_CACHE_DIR =", os.environ["NUMBA_CACHE_DIR"], flush=True)

    sys.path.insert(0, ROOT)
    from robbins_lb import solve, set_below_normal_priority
    from robbins_lb.designs import SPEC_N30, build

    set_below_normal_priority()
    d = build(SPEC_N30)
    print("design:", d.label, flush=True)
    runs = [("down", "f64")] + ([("nearest", "f32")] if a.also_nearest_f32 else [])
    ok = True
    for mode, storage in runs:
        t0 = time.time()
        r = solve(d, mode=mode, storage=storage, threads=a.threads)
        u = float(r["U0"])
        exp = float.fromhex(EXPECTED[(mode, storage)])
        same = u.hex() == exp.hex()
        ok &= same
        print(f"mode={mode} storage={storage}: U0 = {u!r} ({u.hex()})  expected {exp!r}  "
              f"{'MATCH (bit-identical)' if same else 'MISMATCH'}  "
              f"[{r['total_states']} summaries, {time.time() - t0:.0f} s]", flush=True)
    print("QUICK CHECK PASSED" if ok else "QUICK CHECK FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
