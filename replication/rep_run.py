"""Independent replication -- production runs.
python -X utf8 rep_run.py NAME n k th Gf D bc caps(comma) mode storage backend threads
Writes results/NAME.json and results/NAME.log.
"""
import os, sys, json, time, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    name, n, k, th, Gf, D, bc, caps, mode, storage, backend, threads = sys.argv[1:13]
    os.environ["NUMBA_NUM_THREADS"] = threads
    from rep_env import setup_env
    nthr = setup_env(int(threads))
    import numpy as np
    import numba
    import rep_design as RD
    import rep_solver as RS
    caps = None if caps == "none" else [int(c) for c in caps.split(",")]
    d = RD.hybrid(int(n), int(k), float(th), int(Gf), int(D), float(bc), caps)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    logf = open(os.path.join(HERE, "results", name + ".log"), "a", encoding="utf8")

    def log(s):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    work = None
    if backend == "disk":
        work = os.path.join(os.environ["REP_WORKDIR"], "repwork_" + name)
    log(f"start {name}: {d.label} mode={mode} storage={storage} backend={backend} threads={nthr}")
    res = RS.solve(d, mode, storage, backend, work, log)
    import psutil
    res.update(name=name, design=dict(n=d.n, k=d.k, fine=d.fine, coarse=d.coarse, caps=d.caps, label=d.label),
               mode=mode, storage=storage, backend=backend, threads=nthr,
               proc_peak_wset_GiB=round(psutil.Process().memory_info().peak_wset / 2**30, 3),
               numpy=np.__version__, numba=numba.__version__, date=str(datetime.datetime.now()))
    res["layer_sizes"] = {str(k): v for k, v in res["layer_sizes"].items()}
    json.dump(res, open(os.path.join(HERE, "results", name + ".json"), "w"), indent=1)
    log(f"done U0={res['U0']!r} ({res['U0_hex']}) {res['seconds']}s peak_wset={res['proc_peak_wset_GiB']} GiB")


if __name__ == "__main__":
    main()
