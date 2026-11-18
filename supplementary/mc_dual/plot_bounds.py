"""Figure: dual lower bounds and primal upper bounds vs n (from results.json)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
ns = sorted(int(k) for k in R)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for key, lab, mk in [("dual_k2", "dual LB, Ṽ = k2", "v"), ("dual_k3", "dual LB, Ṽ = k3", "v"),
                     ("dual_comb", "dual LB, λ2·M2 + λ3·M3", "^")]:
    ax.errorbar(ns, [R[str(n)][key][0] for n in ns], yerr=[2 * R[str(n)][key][1] for n in ns],
                marker=mk, capsize=3, label=lab)
for key, lab in [("primal_k2", "primal UB, k2 policy"), ("primal_k3", "primal UB, k3 policy")]:
    ax.errorbar(ns, [R[str(n)][key][0] for n in ns], yerr=[2 * R[str(n)][key][1] for n in ns],
                marker="o", ls="--", capsize=3, label=lab)
ax.axhline(1.908, color="k", lw=0.8, ls=":")
ax.text(ns[0], 1.913, "BF 1993 (extrapolated) 1.908", fontsize=8)
ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("n")
ax.set_ylabel("expected rank")
ax.set_title("Robbins' problem: MC dual lower / primal upper bounds on v_n (±2 SE)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "bounds_vs_n.png"), dpi=150)
print("saved")
