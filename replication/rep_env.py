"""thread count + BelowNormal priority for this process only (RULES sec. 4)."""
import os


def setup_env(threads=4):
    import numba
    numba.set_num_threads(int(threads))
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception as e:  # pragma: no cover
        print("warning: could not set priority:", e)
    return numba.get_num_threads()
