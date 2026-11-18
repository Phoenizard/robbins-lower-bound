"""Certified run driver.

Usage (from anywhere; paths are resolved relative to the repository):

    NUMBA_CACHE_DIR=<new empty directory>  python scripts/certify.py RUNID DESIGN [--mode down] [--storage f32]

DESIGN is a named design ("D100" or "D80", see robbins_lb/designs.py) or a JSON spec string such as
'{"n":100,"k":3,"th":7,"Gf":20,"bf":0.0,"D":6,"bc":5.0,"caps":[0.9,0.0,1.0]}'.

The driver refuses to start unless
  (i)   NUMBA_CACHE_DIR is set to an existing, empty directory, so that every numba function is compiled
        from the committed source (numba's cache invalidation tracks only the file of the cached function,
        not inlined helpers from other files);
  (ii)  the output directory runs/RUNID/ does not exist yet, so the checkpoint directory is fresh
        (resumption from a checkpoint is keyed on the design, mode and storage, not on the code version);
  (iii) the git working tree is clean (no modified, staged or untracked files), so that the recorded commit
        identifies the code that runs.
It writes runs/RUNID/RUNID.log (one line per layer) and runs/RUNID/RUNID.cert.json, the certificate record:
design (all edges and caps), mode, storage, repository commit, SHA-256 of every Python file of the
repository that was loaded, versions of Python / NumPy / numba / llvmlite, times, peak working set,
state and cell-integral counts, and the result U0 as repr, hexadecimal, exact fraction and exact decimal.

Only mode "down" gives a rigorous lower bound (Proposition 4.6 of the paper); the record says
"certified": true only for that mode.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git(*a):
    return subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()


def check_preconditions(runid):
    cd = os.environ.get("NUMBA_CACHE_DIR")
    if not cd or not os.path.isdir(cd):
        sys.exit("NUMBA_CACHE_DIR must be set to an existing empty directory")
    if any(fs or ds for _, ds, fs in os.walk(cd)):
        sys.exit(f"NUMBA_CACHE_DIR={cd} is not empty; use a new empty directory")
    if "numba" in sys.modules:
        sys.exit("numba was imported before the cache directory was checked")
    out_dir = os.path.join(ROOT, "runs", runid)
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; every certified run needs a fresh output and checkpoint directory")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        sys.exit("the git working tree is not clean; commit or remove these files first:\n" + dirty)
    return os.path.abspath(cd), out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runid")
    ap.add_argument("design", help='"D100", "D80", or a JSON spec')
    ap.add_argument("--mode", default="down", choices=["down", "nearest"])
    ap.add_argument("--storage", default="f32", choices=["f32", "f64"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem", type=float, default=11.0, help="refuse designs whose two live layers exceed this (GiB)")
    ap.add_argument("--ckpt_every", type=float, default=1800.0, help="seconds between checkpoints")
    a = ap.parse_args()

    cache_dir, out_dir = check_preconditions(a.runid)
    commit = git("rev-parse", "HEAD")

    sys.path.insert(0, ROOT)
    from fractions import Fraction
    from decimal import Decimal, getcontext

    import llvmlite
    import numba
    import numpy as np
    import psutil

    import robbins_lb
    from robbins_lb import solve, set_below_normal_priority
    from robbins_lb import designs

    pkg = os.path.abspath(os.path.dirname(robbins_lb.__file__))
    if pkg != os.path.join(ROOT, "robbins_lb"):
        sys.exit(f"robbins_lb was imported from {pkg}, not from this repository")

    if a.design in ("D100", "D80"):
        spec = getattr(designs, "SPEC_" + a.design)
    else:
        spec = json.loads(a.design)
    d = designs.build(spec)

    set_below_normal_priority()
    os.makedirs(out_dir)
    ckpt_dir = os.path.join(out_dir, "ckpt")
    logp = os.path.join(out_dir, a.runid + ".log")
    lf = open(logp, "a", buffering=1)

    def log(*x):
        s = " ".join(str(i) for i in x)
        lf.write(s + "\n")
        print(s, flush=True)

    log(f"# run {a.runid} start {time.strftime('%Y-%m-%d %H:%M:%S')} mode={a.mode} storage={a.storage} "
        f"threads={a.threads} commit={commit}")
    log("# design:", d.to_dict())
    t0 = time.time()
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    r = solve(d, mode=a.mode, storage=a.storage, threads=a.threads, log=log, mem_limit_gb=a.mem,
              ckpt_dir=ckpt_dir, ckpt_every_s=a.ckpt_every)
    end = time.strftime("%Y-%m-%d %H:%M:%S")
    U0 = float(r["U0"])
    pm = psutil.Process().memory_info()
    peak = getattr(pm, "peak_wset", pm.rss)
    getcontext().prec = 80
    fr = Fraction(U0)

    srcs = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and f.endswith(".py") and os.path.abspath(f).startswith(ROOT + os.sep):
            srcs[os.path.relpath(os.path.abspath(f), ROOT).replace("\\", "/")] = sha(f)
    srcs[os.path.relpath(os.path.abspath(__file__), ROOT).replace("\\", "/")] = sha(__file__)
    srcs = dict(sorted(srcs.items()))
    clean_at_end = git("status", "--porcelain", "--untracked-files=all") == ""

    rec = dict(runid=a.runid, certified=(a.mode == "down"), spec=spec, design=d.to_dict(),
               mode=a.mode, storage=a.storage, threads=a.threads,
               git_commit=commit, git_tree_clean_at_start=True, git_tree_clean_at_end=clean_at_end,
               sha256=srcs,
               python=sys.version, numpy=np.__version__, numba=numba.__version__, llvmlite=llvmlite.__version__,
               platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
               numba_cache_dir=cache_dir, ckpt_dir=ckpt_dir,
               start=start, end=end, wall_seconds=round(time.time() - t0, 1),
               peak_working_set_GiB=round(peak / 2**30, 3), est_peak_layers_GiB=round(r["peak_bytes"] / 2**30, 3),
               U0_repr=repr(U0), U0_hex=U0.hex(), U0_exact_fraction=f"{fr.numerator}/{fr.denominator}",
               U0_exact_decimal=str(Decimal(fr.numerator) / Decimal(fr.denominator)),
               total_states=r["total_states"], max_layer_states=r["max_layer_states"], panels=r["panels"],
               log_file=os.path.basename(logp))
    assert Fraction(Decimal(rec["U0_exact_decimal"])) == fr and float.fromhex(rec["U0_hex"]) == U0
    with open(os.path.join(out_dir, a.runid + ".cert.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    log(f"U0={U0!r} {U0.hex()} exact={rec['U0_exact_decimal']} wall={rec['wall_seconds']}s "
        f"peakWS={rec['peak_working_set_GiB']}GiB")


if __name__ == "__main__":
    main()
