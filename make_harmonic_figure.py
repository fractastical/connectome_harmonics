"""
Generate the illustrative Figure 1 for the paper: connectome harmonics rendered
on the brain (HCP Schaefer-400 layout), from smooth/global (low mode) to
fine-grained (high mode).

Run:
    .venv-lsd/bin/python make_harmonic_figure.py
Output: paper/figs/connectome_harmonic_modes.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "connectome_harmonics_data_hcp.json").read_text())
OUT = ROOT / "paper" / "figs" / "connectome_harmonic_modes.png"

nodes = np.array(DATA["nodes"])          # (400, 2)
modes = np.array(DATA["modes"])          # (40, 400)
eigvals = np.array(DATA["eigvals"])      # (40,)
edges = DATA["edges"]

SHOW = [1, 5, 20]                        # 1-indexed: low, mid, high spatial frequency
LABELS = ["low frequency\n(broad, global)", "mid frequency\n(regional)",
          "high frequency\n(fine-grained)"]

fig, axes = plt.subplots(1, len(SHOW), figsize=(13.5, 4.6))
fig.patch.set_facecolor("#0b0e14")

for ax, m, lab in zip(axes, SHOW, LABELS):
    ax.set_facecolor("#0b0e14")
    ax.set_aspect("equal")
    ax.axis("off")
    for i, j, w in edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]],
                lw=0.15 + 0.5 * w, alpha=0.06, color="white", zorder=1)
    field = modes[m - 1]
    field = field / np.max(np.abs(field))                # normalize for color scale
    ax.scatter(nodes[:, 0], nodes[:, 1], c=field, s=34, cmap="coolwarm",
               vmin=-1, vmax=1, edgecolors="none", zorder=2)
    ax.set_title(f"harmonic mode {m}   (λ = {eigvals[m - 1]:.3f})\n{lab}",
                 color="white", fontsize=11, pad=8)

fig.suptitle(
    "Connectome harmonics on the HCP Schaefer-400 structural connectome "
    "(graph-Laplacian eigenmodes)",
    color="#cfe0f5", fontsize=12.5, y=1.02,
)
fig.tight_layout()
fig.savefig(OUT, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("Wrote", OUT)


# ---------------------------------------------------------------------------
# Decorative, text-free background for the website hero (transparent PNG).
# Two brains (a smooth low mode and a fine high mode) rendered as soft dot
# clouds so the page's gradient shows through behind them.
# ---------------------------------------------------------------------------
BG_OUT = ROOT / "assets" / "hero_harmonics_bg.png"
BG_OUT.parent.mkdir(exist_ok=True)

bg_modes = [2, 18]
bgfig, bgaxes = plt.subplots(1, len(bg_modes), figsize=(16, 7))
bgfig.patch.set_alpha(0.0)
for ax, m in zip(np.atleast_1d(bgaxes), bg_modes):
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")
    for i, j, w in edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]],
                lw=0.2 + 0.7 * w, alpha=0.05, color="#aacbff", zorder=1)
    field = modes[m - 1]
    field = field / np.max(np.abs(field))
    ax.scatter(nodes[:, 0], nodes[:, 1], c=field, s=120, cmap="coolwarm",
               vmin=-1, vmax=1, edgecolors="none", alpha=0.55, zorder=2)

bgfig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
bgfig.savefig(BG_OUT, dpi=130, transparent=True, bbox_inches="tight",
              pad_inches=0)
print("Wrote", BG_OUT)
