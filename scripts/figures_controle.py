"""Figures de contrôle du dataset : spectres de référence et répartition des classes.
Usage : python figures_controle.py --dataset dataset/ --out figures/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, INK, INK2, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

EXEMPLES = [  # (id, titre, valeurs expérimentales de référence en cm-1)
    (577, "Graphène (id 577, P6/mmm)", [(1580, "G exp. ≈1580")]),
    (1, "h-BN monocouche (id 1, P-6m2)", [(1366, "exp. ≈1366")]),
    (2192, "MoS₂ massif 3R (id 2192, R3m)", [(382, "E¹₂g exp. (massif)"), (408, "A₁g exp. (massif)")]),
    (1348, "WS₂ monocouche (id 1348, P-6m2)", []),
]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = np.load(Path(a.dataset) / "spectres_tous.npz", allow_pickle=True)
    X, grid, ids = d["X"], d["grid"], d["id"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), facecolor=SURF)
    for ax, (uid, titre, refs) in zip(axes.ravel(), EXEMPLES):
        y = X[np.flatnonzero(ids == uid)[0]]
        nz = np.flatnonzero(y > 0.02)
        lo, hi = max(100, grid[nz[0]] - 60), min(3200, grid[nz[-1]] + 60)
        if refs:
            lo, hi = min(lo, min(r[0] for r in refs) - 40), max(hi, max(r[0] for r in refs) + 40)
        m = (grid >= lo) & (grid <= hi)
        style(ax)
        ax.plot(grid[m], y[m], color=BLUE, lw=2)
        for k, (x, lab) in enumerate(refs):
            ax.axvline(x, color=INK2, lw=1, ls="--")
            ha = "center" if len(refs) == 1 else ("right" if k == 0 else "left")
            ax.text(x + (0 if ha == "center" else (-2 if ha == "right" else 2)), 1.04, lab, color=INK2, fontsize=8, ha=ha)
        ax.set_ylim(0, 1.12)
        ax.set_title(titre, color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("Décalage Raman (cm⁻¹)", color=INK2, fontsize=9)
        ax.set_ylabel("Intensité normalisée", color=INK2, fontsize=9)
    fig.suptitle("Spectres calculés du dataset (trait plein) et positions expérimentales (tirets)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "spectres_exemples.png", dpi=150, facecolor=SURF)

    meta = pd.read_csv(Path(a.dataset) / "metadata_tous.csv")
    c = meta.famille.value_counts().reindex(["TMD", "carbone", "h-BN", "autre"])
    fig, ax = plt.subplots(figsize=(7, 3), facecolor=SURF)
    style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.8)
    cols = [BLUE, BLUE, BLUE, "#b9b8b2"]
    ax.barh(c.index[::-1], c.values[::-1], color=cols[::-1], height=0.6)
    ax.set_xscale("log")
    for i, v in enumerate(c.values[::-1]):
        ax.text(v * 1.08, i, str(v), va="center", color=INK, fontsize=9)
    ax.set_title("Nombre de matériaux par famille (échelle log)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "repartition_familles.png", dpi=150, facecolor=SURF)


if __name__ == "__main__":
    main()
