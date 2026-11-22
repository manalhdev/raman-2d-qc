"""
Décalage des pics Raman : valeurs THÉORIQUES (Materials Cloud 2024.148, DFT) contre valeurs MESURÉES.

Sorties :
  resultats/decalages_pics.csv
  figures/fig02_decalages_theorique_mesure.(png|svg)

Mesuré :
  - WS2, MoSe2, WSe2 : nos spectres Southampton (pic = maximum local dans une fenêtre, médiane sur les spectres)
  - MoS2 : médiane des positions ajustées du jeu Zenodo 17311728 (monocouche)
  - graphène, h-BN : littérature (pas encore de spectre mesuré dans nos données)

Usage : python scripts/analyse_decalages.py --racine .
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent))
from style_figures import AXE_X_RAMAN, ENCRE_2, TYPE, appliquer_style, sauver  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def pic_mesure(X, grid, lo, hi):
    """Position médiane (cm-1) du maximum dans [lo, hi] sur un ensemble de spectres."""
    w = (grid >= lo) & (grid <= hi)
    pos = []
    for x in X:
        y = x[w]
        if np.isnan(y).any():
            continue
        y = savgol_filter(y - np.percentile(y, 5), 7, 2)
        i = int(np.argmax(y))
        if 0 < i < len(y) - 1:  # vrai maximum local, pas un bord de fenêtre
            pos.append(grid[w][i])
    return float(np.median(pos)), len(pos)


def main(racine):
    racine = Path(racine)
    d = np.load(racine / "data/v3/spectres_reels.npz")
    X, g = d["X"], d["grid"]
    r = pd.read_csv(racine / "data/v3/etiquettes_reels.csv")
    z = pd.read_csv(racine / "data/v1/zenodo_MoS2_mesures.csv")
    z1 = z[z.couche == "1L"]

    ws2 = ((r.source == "Southampton_WS2_2018") & (r.temperature_K == 295)).values
    hs = r.source.str.startswith("Southampton_HS").values

    lignes = []

    def ajoute(mat, mode, theo, id_mc, struct_mc, mes, n, src_mes):
        lignes.append(dict(materiau=mat, mode=mode, theorique_cm1=theo, id_materials_cloud=id_mc,
                           structure_calculee=struct_mc, mesure_cm1=round(mes, 1), n_spectres=n,
                           source_mesure=src_mes))

    # Graphène et h-BN : littérature
    ajoute("Graphène", "G", 1443.4, 577, "monocouche P6/mmm", 1580.0, None,
           "Ferrari & Basko 2013 (littérature)")
    ajoute("h-BN", "E2g", 1356.1, 1, "monocouche P-6m2", 1367.0, None,
           "Gorbachev et al. 2011 (littérature, monocouche)")
    # MoS2 : Zenodo (monocouche mesurée) ; attention, le calcul disponible est du MoS2 massif 3R
    ajoute("MoS₂", "E¹₂g / E′", 363.5, 2192, "massif 3R (pas de monocouche dans la base)",
           float(z1.E2g_cm1.median()), len(z1), "Zenodo 17311728 (monocouche)")
    ajoute("MoS₂", "A₁g / A₁′", 396.5, 2192, "massif 3R (pas de monocouche dans la base)",
           float(z1.A1g_cm1.median()), len(z1), "Zenodo 17311728 (monocouche)")
    # WS2 : Southampton D0371, 295 K
    m, n = pic_mesure(X[ws2], g, 340, 365)
    ajoute("WS₂", "E′ (+2LA)", 339.6, 1348, "monocouche P-6m2", m, n, "Southampton D0371 (monocouche, 295 K)")
    m, n = pic_mesure(X[ws2], g, 405, 430)
    ajoute("WS₂", "A₁′", 402.7, 1348, "monocouche P-6m2", m, n, "Southampton D0371 (monocouche, 295 K)")
    # MoSe2 / WSe2 : hétérobicouche Southampton D1747, 4 K
    m, n = pic_mesure(X[hs], g, 232, 245)
    ajoute("MoSe₂", "A₁′", 233.3, 168, "monocouche P-6m2", m, n, "Southampton D1747 (hétérobicouche, 4 K)")
    m, n = pic_mesure(X[hs], g, 280, 296)
    ajoute("MoSe₂", "E′", 274.6, 168, "monocouche P-6m2", m, n, "Southampton D1747 (hétérobicouche, 4 K)")
    m, n = pic_mesure(X[hs], g, 245, 256)
    ajoute("WSe₂", "A₁′ / E′", 238.7, 291, "monocouche P-6m2", m, n, "Southampton D1747 (hétérobicouche, 4 K)")

    df = pd.DataFrame(lignes)
    df["decalage_cm1"] = (df.mesure_cm1 - df.theorique_cm1).round(1)
    df["decalage_pct"] = (100 * df.decalage_cm1 / df.mesure_cm1).round(1)
    (racine / "resultats").mkdir(exist_ok=True)
    df.to_csv(racine / "resultats/decalages_pics.csv", index=False)
    print(df[["materiau", "mode", "theorique_cm1", "mesure_cm1", "decalage_cm1", "decalage_pct", "n_spectres"]]
          .to_string(index=False))

    # Figure : un point théorique et un point mesuré par mode, reliés ; décalage écrit à droite
    appliquer_style()
    df_ = df.iloc[::-1].reset_index(drop=True)
    noms = [f"{a} — {b}" for a, b in zip(df_.materiau, df_["mode"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.05),
                             sharey=True)
    for ax, (lo, hi) in zip(axes, [(200, 450), (1330, 1610)]):
        for i, row in df_.iterrows():
            if not (lo <= row.mesure_cm1 <= hi):
                continue
            ax.plot([row.theorique_cm1, row.mesure_cm1], [i, i], color="#c9c8c2", lw=2, zorder=1)
            ax.scatter(row.theorique_cm1, i, s=60, color=TYPE["théorique"], zorder=2,
                       edgecolor="white", linewidth=1.5)
            ax.scatter(row.mesure_cm1, i, s=60, color=TYPE["mesuré"], zorder=2, edgecolor="white", linewidth=1.5)
            x_txt = max(row.theorique_cm1, row.mesure_cm1) + (hi - lo) * 0.025
            ax.text(x_txt, i, f"{row.decalage_cm1:+.0f} cm⁻¹", va="center", fontsize=9.5, color=ENCRE_2)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(AXE_X_RAMAN)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(noms)), noms)
    axes[1].tick_params(left=False)
    axes[0].scatter([], [], color=TYPE["théorique"], s=60, label="Théorique (Materials Cloud, DFT)")
    axes[0].scatter([], [], color=TYPE["mesuré"], s=60, label="Mesuré")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
    fig.suptitle("Les fréquences calculées sont toutes trop basses : de 7 à 137 cm⁻¹ sous la mesure (1 à 9 %)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    sauver(fig, racine / "figures/fig02_decalages_theorique_mesure")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--racine", default=".")
    main(p.parse_args().racine)
