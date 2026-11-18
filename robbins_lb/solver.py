"""Backward pass of the hybrid relaxation, eqs. (7)-(8): U_0 <= v_n (Corollary 3.8; in "down" mode the
returned number is <= U_0 by Proposition 4.6)."""
from __future__ import annotations

import json
import os
import time

import numba as nb
import numpy as np

from .design import Design
from .encoding import layer_encoding, memory_tables
from .kernel import closed_form_A, closed_form_B, layer_kernel


def _cell_widths(edges, down):
    h = np.diff(np.asarray(edges, dtype=np.float64))
    if down:
        h = np.nextafter(h, -np.inf)       # hd <= exact width (one subtraction, rounded to nearest, then dn)
    return h


def set_below_normal_priority():
    """Lower the priority of this process (BelowNormal on Windows, nice 10 elsewhere); errors are ignored."""
    try:
        import psutil
        p = psutil.Process()
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception:
        pass


def solve(design: Design, mode: str = "nearest", storage: str = "f64", threads: int = 8,
          verbose: bool = False, log=None, mem_limit_gb: float = 12.0, ckpt_dir: str | None = None,
          ckpt_every_s: float = 1200.0):
    """Return a dict with U0 (float), timings, state counts and memory.

    mode    : "nearest" (float exploration) or "down" (every stored value is a proven lower bound).
    storage : "f64" or "f32" (layer arrays; "f32" in mode "down" rounds stored values down).
    ckpt_dir: if given, the newest completed layer is saved there every ckpt_every_s seconds and a run with the
              same design/mode/storage resumes from it.
    """
    if mode not in ("nearest", "down"):
        raise ValueError(mode)
    if storage not in ("f64", "f32"):
        raise ValueError(storage)
    down = mode == "down"
    store32 = storage == "f32"
    dtype = np.float32 if store32 else np.float64
    nb.set_num_threads(min(threads, nb.config.NUMBA_NUM_THREADS))
    say = log or (print if verbose else (lambda *a, **k: None))

    d = design
    n, k = d.n, d.k
    peak = d.peak_bytes(storage)
    if peak > mem_limit_gb * 2**30:
        raise MemoryError(f"design needs {peak/2**30:.2f} GiB > limit {mem_limit_gb} GiB")
    f = np.asarray(d.f, dtype=np.float64)
    E = np.asarray(d.E, dtype=np.float64)
    hf = _cell_widths(f, down) if down else np.diff(f)
    hE = _cell_widths(E, down) if down else np.diff(E)
    mt = memory_tables(k, d.Gf)
    bs = mt["bs"]
    A = closed_form_A(down, hf, mt["cnt"])
    caps = np.array(d.caps if d.caps is not None else [n + 1] * d.D, dtype=np.int64)
    t0 = time.time()

    if n == 1:
        U0 = float(closed_form_B(down, hE, np.zeros(d.D, dtype=np.int64)))
        U0 = float(np.nextafter(A[0] + U0, -np.inf)) if down else float(A[0] + U0)
        return dict(U0=U0, mode=mode, storage=storage, seconds=time.time() - t0, max_layer_states=1,
                    total_states=1, peak_bytes=0, panels=0, design=d.to_dict())

    # ---- resume
    start_t = n - 2
    Unext = np.zeros(1, dtype=dtype)
    meta_key = dict(design=d.to_dict(), mode=mode, storage=storage)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        mp = os.path.join(ckpt_dir, "meta.json")
        if os.path.exists(mp):
            meta = json.load(open(mp))
            if meta.get("key") == meta_key and not meta.get("done"):
                Unext = np.load(os.path.join(ckpt_dir, "layer.npy"))
                start_t = meta["t"] - 1
                say(f"resumed from checkpoint layer t={meta['t']}")
    last_ck = time.time()

    max_states = 0
    tot_states = 0
    panels = 0
    layer_log = []
    ubn = None
    enc_next = None
    if start_t < n - 2:
        enc_next = layer_encoding(d.ub(start_t + 1), k, bs)
    for t in range(start_t, -1, -1):
        tl = time.time()
        ub = np.array(d.ub(t), dtype=np.int64)
        cnt, pref, base, ns = layer_encoding(ub, k, bs)
        cf = (t == n - 2)
        if cf:
            cntn, prefn, basen = cnt, pref, base          # unused
        else:
            cntn, prefn, basen, _ = enc_next
        Ucur = np.empty(ns, dtype=dtype)
        nseq = int(pref[0, ub[0] + 1])
        chunk = int(max(1, min(4096, nseq // (64 * max(1, threads)))))
        layer_kernel(down, store32, cf, n, t, k, f, E, hf, hE, caps,
                     ub, cnt, pref, base, bs, cntn, prefn, basen,
                     mt["bstart"], mt["nxt"], mt["nxtloc"], mt["cnt"], A,
                     Unext, Ucur, np.int64(chunk))
        Unext = Ucur
        enc_next = (cnt, pref, base, ns)
        max_states = max(max_states, ns)
        tot_states += ns
        panels += ns * (d.Gf + d.D)
        layer_log.append((t, ns, time.time() - tl))
        say(f"  t={t:3d} states={ns:.4g} {time.time()-tl:.1f}s")
        if ckpt_dir and time.time() - last_ck > ckpt_every_s and t > 0:
            tmp = os.path.join(ckpt_dir, "layer.tmp.npy")
            np.save(tmp, Ucur)
            os.replace(tmp, os.path.join(ckpt_dir, "layer.npy"))
            json.dump(dict(key=meta_key, t=t), open(os.path.join(ckpt_dir, "meta.json"), "w"))
            last_ck = time.time()
    # initial state: S = 0, K empty -> block 0, local 0 -> flat index 0
    U0 = float(Unext[0])
    out = dict(U0=U0, mode=mode, storage=storage, seconds=time.time() - t0,
               max_layer_states=int(max_states), total_states=int(tot_states),
               last_layer_states_closed_form=int(d.layer_states(n - 1)),
               peak_bytes=int(peak), panels=int(panels), threads=threads, design=d.to_dict())
    if ckpt_dir:
        json.dump(dict(key=meta_key, t=0, done=True, U0=U0), open(os.path.join(ckpt_dir, "meta.json"), "w"))
    return out
