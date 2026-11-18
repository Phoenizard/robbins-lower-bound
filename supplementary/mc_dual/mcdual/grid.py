"""Non-uniform grid g_0=0 < g_1 < ... < g_{G-1}=1, g(u) = (exp(beta u)-1)/(exp(beta)-1), u = i/(G-1)."""
import math
import numpy as np


def make_grid(G: int, beta: float) -> np.ndarray:
    u = np.arange(G) / (G - 1)
    if beta == 0.0:
        g = u.copy()
    else:
        g = np.expm1(beta * u) / math.expm1(beta)
    g[0] = 0.0
    g[-1] = 1.0
    return g


def default_beta(n: int) -> float:
    """Concentrate nodes where stopping thresholds live (x ~ 1/n)."""
    return max(1.0, math.log(n) + 2.0)
