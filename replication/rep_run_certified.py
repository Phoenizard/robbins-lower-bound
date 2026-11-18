"""Independent replication of the certified design n100_g20_down_f32 using its EXACT binary64 edges and caps
(read as data from certificates/original/n100_g20_down_f32.cert.json; the replication's own edge formula gives
the same caps and edges equal up to 1-3 ulp, see README.md).
python -X utf8 rep_run_certified.py NAME mode storage backend threads
"""
import os, sys, json, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CERT = os.path.join(HERE, "..", "certificates", "original", "n100_g20_down_f32.cert.json")


def main():
    name, mode, storage, backend, threads = sys.argv[1:6]
    os.environ["NUMBA_NUM_THREADS"] = threads
    from rep_env import setup_env
    nthr = setup_env(int(threads))
    import numpy as np, numba, psutil
    import rep_design as RD
    import rep_solver as RS
    c = json.load(open(CERT))["design"]
    d = RD.custom(c["n"], c["k"], c["f"], c["E"], c["caps"], label="cert " + c["label"])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    logf = open(os.path.join(HERE, "results", name + ".log"), "a", encoding="utf8")

    def log(s):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    work = os.path.join(os.environ["REP_WORKDIR"], "repwork_" + name) if backend == "disk" else None
    log(f"start {name}: {d.label} mode={mode} storage={storage} backend={backend} threads={nthr}")
    res = RS.solve(d, mode, storage, backend, work, log)
    from fractions import Fraction
    res.update(name=name, design=dict(n=d.n, k=d.k, fine=d.fine, coarse=d.coarse, caps=d.caps, label=d.label),
               mode=mode, storage=storage, backend=backend, threads=nthr,
               U0_exact_fraction=str(Fraction(res["U0"])),
               proc_peak_wset_GiB=round(psutil.Process().memory_info().peak_wset / 2**30, 3),
               numpy=np.__version__, numba=numba.__version__, date=str(datetime.datetime.now()))
    res["layer_sizes"] = {str(k): v for k, v in res["layer_sizes"].items()}
    json.dump(res, open(os.path.join(HERE, "results", name + ".json"), "w"), indent=1)
    log(f"done U0={res['U0']!r} ({res['U0_hex']}) {res['seconds']}s peak_wset={res['proc_peak_wset_GiB']} GiB")


if __name__ == "__main__":
    main()
