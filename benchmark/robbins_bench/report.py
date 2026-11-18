"""Print markdown summary tables from results/*.json (used to write the benchmark README).

    python -I -X utf8 -m robbins_bench.report      (run from benchmark/)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

BF = {
    1: {2: "1.2500", 3: "1.3242", 5: "1.3803", 9: "1.4171", 17: "1.4385"},
    2: {3: "1.3915", 5: "1.4991", 9: "1.5689", 17: "1.6092"},
    3: {5: "1.5509", 9: "1.6490", 17: "1.7070"},
    4: {5: "1.5710", 9: "1.6956", 17: "1.7668"},
    5: {9: "1.7260", 17: "1.8090"},
}


def load(name):
    p = os.path.join(RESULTS, name + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def extrapolate(points):
    """points: list of (G, mid). Fit mid = W + c2/G^2 (+ c3/G^3) on the finest points. Non-rigorous."""
    pts = sorted(points)
    G = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    ests = []
    for deg, k in ((2, 2), (2, 3), (3, 3), (3, 4)):
        if len(G) < k:
            continue
        g, v = G[-k:], y[-k:]
        cols = [np.ones_like(g)] + [g ** -e for e in range(2, deg + 1)]
        A = np.vstack(cols).T
        sol, *_ = np.linalg.lstsq(A, v, rcond=None)
        ests.append(sol[0])
    return float(np.median(ests)), float(np.max(ests) - np.min(ests)) if len(ests) > 1 else float("nan")


def main():
    print("## Rigorous enclosures of W_n^(m) (BF Table 3 entries)\n")
    print("| m | G | n | lower | upper | width | BF printed | inside? |")
    print("|---|---|---|---|---|---|---|---|")
    series = load("series")
    for m in range(1, 6):
        r = load(f"m{m}")
        if r is None:
            continue
        for n, s in BF[m].items():
            lo, hi = r["W_lower"][str(n)], r["W_upper"][str(n)]
            v = float(s)
            ins = "yes" if lo - 5e-5 <= v <= hi + 5e-5 else "**no**"
            print(f"| {m} | {r['G']} | {n} | {lo:.6f} | {hi:.6f} | {hi - lo:.6f} | {s} | {ins} |")
    print("\n## Non-rigorous extrapolation of midpoints (L+U)/2 in G\n")
    print("| m | n | estimate | spread of fits | BF printed |")
    print("|---|---|---|---|---|")
    for m in range(1, 6):
        r = load(f"m{m}")
        pts = {}
        if series:
            for key, run in series["runs"].items():
                if run["m"] == m:
                    for n in BF[m]:
                        pts.setdefault(n, []).append((run["G"], 0.5 * (run["lower"][n - 1] + run["upper"][n - 1])))
        if r is not None:
            for n in BF[m]:
                pts.setdefault(n, []).append((r["G"], 0.5 * (r["W_lower"][str(n)] + r["W_upper"][str(n)])))
        for n, s in BF[m].items():
            if len(pts.get(n, [])) >= 2:
                est, spr = extrapolate(pts[n])
                print(f"| {m} | {n} | {est:.5f} | {spr:.1e} | {s} |")
            elif r is not None:
                mid = 0.5 * (r["W_lower"][str(n)] + r["W_upper"][str(n)])
                print(f"| {m} | {n} | {mid:.5f} (midpoint, G={r['G']}) | - | {s} |")
    print("\n## All n (production grids)\n")
    print("| n | " + " | ".join(f"m={m}" for m in range(1, 6)) + " |")
    print("|---|" + "---|" * 5)
    rs = {m: load(f"m{m}") for m in range(1, 6)}
    for n in range(1, 18):
        cells = []
        for m in range(1, 6):
            r = rs[m]
            cells.append("-" if r is None else f"[{r['W_lower'][str(n)]:.5f}, {r['W_upper'][str(n)]:.5f}]")
        print(f"| {n} | " + " | ".join(cells) + " |")
    kb = load("kbest")
    if kb:
        print("\n## Brice et al. (d,k)-abstraction\n")
        print("| n | d | k | published | ours (code variant) | ours (Def. 5 as printed) | match |")
        print("|---|---|---|---|---|---|---|")
        for c in kb["cases"]:
            pub = c["published"]
            ulp = 10 ** (-len(pub.split(".")[1]))
            diff = abs(c["code_variant"] - float(pub))
            tag = "yes (rounds)" if diff <= 0.5 * ulp + 1e-12 else ("yes (1 ulp, truncated?)" if diff <= ulp else "no")
            print(f"| {c['n']} | {c['d']} | {c['k']} | {pub} | {c['code_variant']:.9f} | "
                  f"{c['paper_variant']:.9f} | {tag} |")


if __name__ == "__main__":
    main()
