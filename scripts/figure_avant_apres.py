"""Figure avant/après prétraitement, pour le rapport (to-do item 2).

Montre, sur un vrai spectre mesuré (WS2, Southampton), les étapes du prétraitement final :
brut -> correction de ligne de base (ALS) -> débruitage (Savitzky-Golay) -> normalisé
(sans le pic du silicium).

Usage : python scripts/figure_avant_apres.py --v3 data/v3/ --out figures/
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pretraitement import ZONE_SILICIUM, corriger_ligne_base, debruiter

BLUE, ORANGE, INK, INK2, GRID_C, SURF = "#2a78d6", "#d6692a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID_C, lw=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="Southampton_WS2_2018")
    a = ap.parse_args()

    d = np.load(Path(a.v3) / "spectres_reels.npz")
    X, grid = d["X"], d["grid"]
    lab = pd.read_csv(Path(a.v3) / "etiquettes_reels.csv")

    idx = np.flatnonzero(lab.source == a.source)
    i = idx[len(idx) // 2]
    row = lab.iloc[i]
    y = X[i]

    y_base = corriger_ligne_base(X[i:i + 1], grid, lam=1e5, p=0.01, niter=10)[0]
    y_denoise = debruiter(y_base[None, :], fenetre=11, ordre=3)[0]

    hors_si = ~((grid >= ZONE_SILICIUM[0]) & (grid <= ZONE_SILICIUM[1]))
    m = np.nanmax(np.where(hors_si, y_denoise, np.nan))
    y_norm = y_denoise / (m if np.isfinite(m) and m > 0 else 1.0)

    ok = np.isfinite(y)
    xr = grid[ok]
    xmax = xr.max()
    m_affiche = (grid >= xr.min()) & (grid <= xmax)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), facecolor=SURF, sharex=True)

    ax = axes[0]
    style(ax)
    ax.plot(grid[m_affiche], y[m_affiche], color=INK, lw=1.0, label="brut (mesuré)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylabel("Intensité\n(unités de la source)", color=INK2, fontsize=8)
    ax.set_title(
        f"Avant / après prétraitement — {row.materiau} ({row.structure}), "
        f"{row.temperature_K:.0f} K, laser {row.excitation_nm:.0f} nm",
        color=INK, fontsize=11, loc="left",
    )

    ax = axes[1]
    style(ax)
    ax.plot(grid[m_affiche], y[m_affiche], color="#c9c7c2", lw=1.0, label="brut")
    ax.plot(grid[m_affiche], y_denoise[m_affiche], color=ORANGE, lw=1.2,
            label="ligne de base ALS retirée + débruité (Savitzky-Golay)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylabel("Intensité\n(unités de la source)", color=INK2, fontsize=8)

    ax = axes[2]
    style(ax)
    ax.axvspan(ZONE_SILICIUM[0], ZONE_SILICIUM[1], color="#eee3cf", lw=0,
               label="pic du substrat Si (ignoré pour la normalisation)")
    ax.plot(grid[m_affiche], y_norm[m_affiche], color=BLUE, lw=1.2, label="prétraité et normalisé")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylabel("Intensité\n(normalisée, sans unité)", color=INK2, fontsize=8)
    ax.set_xlabel("Décalage Raman (cm$^{-1}$)", color=INK2, fontsize=9)

    fig.tight_layout()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out) / "avant_apres_pretraitement.png"
    fig.savefig(out_path, dpi=150, facecolor=SURF)
    print("Figure enregistrée :", out_path)


if __name__ == "__main__":
    main()
