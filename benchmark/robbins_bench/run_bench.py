"""Production runs of the validation anchor.  Usage (from benchmark/):

    python -I -X utf8 run.py <task> [<task> ...]

tasks:  m1 m2 m3 m4 m5   rigorous enclosures of W_n^(m), n = 1..17 (largest grid used)
        series           midpoint series over several G for m = 3, 4, 5 (non-rigorous extrapolation)
        exact            v_1..v_4 exact / high precision; m = 1 column by the independent solver
        policysim        Monte Carlo of an explicit rule under loss (4.4): upper check on W
        kbest            Brice et al. (d,k)-abstraction values (Table 1 rows n = 5, 10; n = 3, 4 checks)
Each task writes results/<task>.json.  The process lowers its own priority to BELOW_NORMAL (Windows).
"""
import ctypes
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

# rigorous production grids: (m, G, p) -- power grid t_i = (i/G)^p, dyadic
PRODUCTION = {
    1: (1, 2 ** 18, 1.5),
    2: (2, 8192, 1.5),
    3: (3, 768, 1.5),
    4: (4, 224, 1.5),
    5: (5, 104, 1.5),
}
SERIES = {3: [64, 96, 128, 192, 256], 4: [32, 48, 64, 96, 128], 5: [24, 32, 40, 48, 64]}
NMAX = 17


def below_normal():
    if platform.system() == "Windows":
        try:
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception as e:  # pragma: no cover
            print("could not lower priority:", e)


def meta():
    import mpmath
    import numba
    return dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0],
                numpy=np.__version__, numba=numba.__version__, mpmath=mpmath.__version__,
                machine=platform.machine())


def dump(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def task_enclose(m):
    from .grid import make_grid
    from .truncated_dp import enclose_W, n_states
    _, G, p = PRODUCTION[m]
    t = make_grid(G, "power", p)
    t0 = time.time()
    lo = enclose_W(m, t, NMAX, True, verbose=True)
    t1 = time.time()
    hi = enclose_W(m, t, NMAX, False, verbose=True)
    t2 = time.time()
    dump(f"m{m}", dict(meta=meta(), m=m, G=G, grid=f"power p={p}", states=n_states(G, m),
                       seconds_lower=t1 - t0, seconds_upper=t2 - t1,
                       W_lower={n: lo[n - 1] for n in range(1, NMAX + 1)},
                       W_upper={n: hi[n - 1] for n in range(1, NMAX + 1)},
                       note="rigorous: W_lower[n] <= W_n^(m) <= W_upper[n] (monotone discretisation + "
                            "directed rounding)"))


def task_series():
    from .grid import make_grid
    from .truncated_dp import enclose_W_both
    out = dict(meta=meta(), grid="power p=1.5", runs={})
    for m, Gs in SERIES.items():
        for G in Gs:
            t0 = time.time()
            lo, hi = enclose_W_both(m, make_grid(G, "power", 1.5), NMAX)
            out["runs"][f"m{m}_G{G}"] = dict(m=m, G=G, lower=lo, upper=hi, seconds=time.time() - t0)
            print(m, G, lo[16], hi[16], flush=True)
            dump("series", out)
    dump("series", out)


def task_exact():
    import mpmath as mp

    from . import m1_exact, smalln
    out = dict(meta=meta())
    v3, v3n, root, _ = smalln.v3_exact()
    out["v1"] = "1"
    out["v2"] = str(smalln.v2_exact())
    out["v3_exact"] = str(v3)
    out["v3_50digits"] = str(v3n)
    out["v3_threshold_first_obs"] = str(root)
    for dps in (30, 40):
        t0 = time.time()
        v4, s = smalln.v4_highprec(dps)
        out[f"v4_dps{dps}"] = mp.nstr(v4, dps)
        out[f"v4_threshold_dps{dps}"] = mp.nstr(s, dps)
        out[f"v4_seconds_dps{dps}"] = time.time() - t0
        print("v4", dps, mp.nstr(v4, dps), flush=True)
    for dps in (60, 90):
        w = m1_exact.W1_values(NMAX, dps)
        out[f"W1_dps{dps}"] = {n: mp.nstr(w[n - 1], dps - 15) for n in range(1, NMAX + 1)}
    dump("exact", out)


KBEST_CASES = [
    # (n, d, k, published value, source)
    (3, 500, 3, "1.391635988", "Sec 6.1 v_{3,500}"),
    (3, 1000, 3, "1.391593893", "Sec 6.1 v_{3,1000}"),
    (4, 500, 2, "1.49433", "Sec 6.2 v_{4,500,2}"),
    (4, 1000, 2, "1.49423", "Sec 6.2 v_{4,1000,2}"),
    (5, 100, 2, "1.57347", "Table 1"),
    (5, 100, 3, "1.57219", "Table 1"),
    (5, 500, 2, "1.57231", "Table 1"),
    (5, 500, 3, "1.57128", "Table 1"),
    (5, 1000, 2, "1.57217", "Table 1"),
    (10, 100, 2, "1.80133", "Table 1"),
    (10, 100, 3, "1.79654", "Table 1"),
    (10, 500, 2, "1.79912", "Table 1"),
    (10, 500, 3, "1.79463", "Table 1"),
    (10, 1000, 2, "1.79886", "Table 1"),
]


def task_kbest():
    import numba
    numba.set_num_threads(int(os.environ.get("BENCH_THREADS", "1")))
    from .kbest import kbest_value
    out = dict(meta=meta(), cases=[])
    for n, d, k, pub, src in KBEST_CASES:
        t0 = time.time()
        a = kbest_value(n, d, k, "code")
        b = kbest_value(n, d, k, "paper")
        ulp = 10 ** (-(len(pub.split(".")[1])))
        rec = dict(n=n, d=d, k=k, published=pub, source=src, code_variant=a, paper_variant=b,
                   code_matches=abs(a - float(pub)) <= ulp + 1e-12,              # within 1 unit of last digit
                   code_rounds_to_published=abs(a - float(pub)) <= 0.5 * ulp + 1e-12,
                   seconds=time.time() - t0)
        out["cases"].append(rec)
        print(rec, flush=True)
        dump("kbest", out)
    dump("kbest", out)


def main(argv):
    below_normal()
    for task in argv:
        print("=== task", task, flush=True)
        if task in ("m1", "m2", "m3", "m4", "m5"):
            task_enclose(int(task[1]))
        elif task == "series":
            task_series()
        elif task == "exact":
            task_exact()
        elif task == "kbest":
            task_kbest()
        elif task == "policysim":
            from .run_policy_sim import run
            dump("policysim", dict(meta=meta(), cases=run(),
                                   note="E[Y_tau] of an explicit stopping rule under loss (4.4): >= W_n^(m)"))
        else:
            raise SystemExit("unknown task " + task)


if __name__ == "__main__":
    main(sys.argv[1:])
