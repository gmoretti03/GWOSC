"""Summary figures of the report (extension 64): rho50 of every method on one axis and the pipeline diagram.
Reads only light results (results/06_summary.json, 07_summary.json, 08_summary.json); run: python make_summary_figures.py"""
import json, os, sys
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwosc_tools as g
g.style()
S6, S7, S8 = (g.load_json(f"{n}_summary.json") for n in ("06", "07", "08"))

def mean3(rows, key, name, far):
    v = [r["rho50"] for r in rows if r[key] == name and r["FAP"] == far and r["morphology"] in g.MORPHS]
    return float(np.mean(v)) if len(v) == 3 else np.nan

three = [("matched-filter bank", S6["detection"], "statistic", "matched-filter bank (SG)", "freq"),
         ("Bayes factor (3 families)", S6["detection"], "statistic", "Bayes (3 families)", "bayes"),
         ("CNN ensemble, static (nb05)", S7["detection"], "model", "CNN ensemble (nb05, 5)", "ml"),
         ("CNN, fresh injections (nb07)", S7["detection"], "model", "cnn_fresh s0", "ml"),
         ("hybrid conv+attention, fresh", S7["detection"], "model", "hyb_fresh s0", "ml2"),
         ("Transformer (nb06)", S7["detection"], "model", "Transformer (nb06, mean of 2)", "ml2")]
imr = [("oracle", "oracle (exact waveform known)", "k"), ("IMR bank", "IMR matched-filter bank", "freq"),
       ("CNN trained with IMR", "CNN fresh + IMR (2 seeds)", "ml"), ("CNN without IMR", "CNN fresh (nb07, 2 seeds)", "ml"),
       ("sine-Gaussian bank", "sine-Gaussian bank", "freq"), ("hybrid, no IMR", "hybrid fresh (nb07)", "ml2"), ("excess power", "excess power", "grey")]
col = dict(freq=g.COLORS["freq"], bayes=g.COLORS["bayes"], ml=g.COLORS.get("ml", "C2"), ml2="C4", k="k", grey=g.COLORS["grey"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for j, far in enumerate((1e-2, 1e-3)):
    vals = [(lab, mean3(rows, key, name, far), c) for lab, rows, key, name, c in three]
    for i, (lab, v, c) in enumerate(vals):
        ax[0].barh(i + 0.2 * (1 - 2 * j) / 1.0 * 0.9 - 0.0, v, height=0.35, color=col[c], alpha=1.0 if j == 0 else 0.45)
    if j == 0: ax[0].set_yticks(range(len(vals))); ax[0].set_yticklabels([v[0] for v in vals])
ax[0].set_xlabel("mean ρ₅₀ over sg / pulse / chirp  (solid: FAP 1e-2, light: 1e-3)"); ax[0].invert_yaxis(); ax[0].set_title("Phenomenological bursts (validation thresholds)")
for j, far in enumerate((1e-2, 1e-3)):
    for i, (lab, name, c) in enumerate(imr):
        r = [x["rho50"] for x in S8["detection"] if x["statistic"] == name and x["FAP"] == far]
        if r: ax[1].barh(i + 0.2 * (1 - 2 * j) * 0.9, r[0], height=0.35, color=col[c], alpha=1.0 if j == 0 else 0.45)
    if j == 0: ax[1].set_yticks(range(len(imr))); ax[1].set_yticklabels([x[0] for x in imr])
ax[1].set_xlabel("ρ₅₀ for IMRPhenomD signals"); ax[1].invert_yaxis(); ax[1].set_title("Binary black holes (held-out library)")
fig.tight_layout(); g.savefig(fig, "fig13_summary_rho50"); plt.close(fig)

# ---- pipeline diagram
fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(-0.2, 3.6)
def box(x, y, w, h, txt, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="k", lw=0.8, alpha=0.85))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8)
def arrow(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1))
box(0.1, 1.4, 1.6, 0.9, "GWOSC O1 H1\n8 × 4096 s chunks\n(+ H1/L1 for nb 09)", "#dddddd")
box(2.1, 1.4, 1.9, 0.9, "whitening (median Welch)\n1 s windows, 30–500 Hz\nsplit by chunk (01)", "#dddddd")
box(4.3, 2.5, 2.2, 0.8, "frequency-domain injections\nsg · pulse · chirp · IMR (02, 08)", "#f4d9a6")
box(4.3, 0.3, 2.2, 0.8, "real background\nthresholds at given FAP", "#f4d9a6")
box(6.9, 2.4, 2.3, 1.0, "frequentist (03)\nbank, profile likelihood", col["freq"])
box(6.9, 1.2, 2.3, 1.0, "Bayesian (04)\ngrid posterior, MCMC, BF", col["bayes"])
box(6.9, 0.0, 2.3, 1.0, "machine learning (05–07)\nCNN, Transformer, AE", col["ml"])
box(9.6, 2.0, 2.2, 1.4, "extensions\nIMR bank (08)\nH1+L1 coincidence (09)\nopen set (10)\nbootstrap, thresholds (11–12)", "#cfe3f5")
box(9.6, 0.3, 2.2, 1.1, "comparison at equal FAP\nρ₅₀, ρ₉₀, AUC\n(report sections 8–10)", "#ffffff")
arrow(1.7, 1.85, 2.1, 1.85); arrow(4.0, 1.85, 4.3, 2.9); arrow(4.0, 1.85, 4.3, 0.7)
for y in (2.9, 1.7, 0.5): arrow(6.5, 2.9 if y > 2 else 0.7, 6.9, y)
arrow(9.2, 2.9, 9.6, 2.7); arrow(9.2, 0.5, 9.6, 0.8); arrow(9.2, 1.7, 9.6, 1.0)
g.savefig(fig, "fig13_pipeline"); plt.close(fig)
print("saved fig13_summary_rho50, fig13_pipeline")
