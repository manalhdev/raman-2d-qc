"""
Figures « données » du rapport et de la présentation (style commun : scripts/style_figures.py).

  fig01_materiaux_par_famille     combien de MATÉRIAUX différents par famille (et pas seulement de spectres)
  fig03_ws2_theorique_simule_mesure  le même matériau vu par le calcul, la simulation et la mesure
  fig09_exemples_simules           un spectre simulé par famille (remplace exemples_v2.png)
  fig10_exemples_mesures           trois spectres mesurés (remplace spectres_reels_v3.png)

Usage : python scripts/figures_donnees.py --racine .
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from style_figures import (AXE_X_RAMAN, AXE_Y_RAMAN, BLEU, COULEURS, ENCRE_2, ORANGE, VERT_EAU,  # noqa: E402
                           appliquer_style, sauver)

import matplotlib.pyplot as plt  # noqa: E402


def norm(y, grid=None):
    y = np.asarray(y, float)
    y = y - np.nanpercentile(y, 1)
    if grid is not None:  # maximum hors silicium
        hors = ~((grid >= 505) & (grid <= 535))
        return y / np.nanmax(np.where(hors, y, np.nan))
    return y / np.nanmax(y)


def fig01(racine, lab):
    cv = lab[lab.split == "cv"]
    re = lab[lab.split == "test_reel"]
    fam = ["carbone", "h-BN", "TMD"]
    n_sim = [cv[cv.famille == f].groupe.nunique() for f in fam]
    n_spec = [len(cv[cv.famille == f]) for f in fam]
    n_mes = [re[re.famille == f].source.nunique() for f in fam]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(fam))
    h = 0.36
    ax.barh(y + h / 2, n_sim, height=h, color=BLEU, label="Matériaux calculés (simulés)")
    ax.barh(y - h / 2, n_mes, height=h, color=ORANGE, label="Matériaux mesurés")
    for i in range(len(fam)):
        ax.text(n_sim[i] + 1.5, y[i] + h / 2, f"{n_sim[i]} matériaux · {n_spec[i]} spectres", va="center",
                fontsize=10, color=ENCRE_2)
        ax.text(n_mes[i] + 1.5, y[i] - h / 2, f"{n_mes[i]}", va="center", fontsize=10, color=ENCRE_2)
    ax.set_yticks(y, fam)
    ax.invert_yaxis()
    ax.set_xlabel("Nombre de matériaux différents")
    ax.set_xlim(0, 215)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Autant de spectres par famille, mais 3 matériaux seulement en carbone et en h-BN")
    return sauver(fig, racine / "figures/fig01_materiaux_par_famille")


def fig03(racine, lab, X, grid):
    v1 = pd.read_csv(racine / "data/v1/spectres_3familles.csv")
    cols = [c for c in v1.columns if c.isdigit()]
    g1 = np.array(cols, float)
    theo = v1[v1.id == 1348][cols].values[0]
    sim_i = np.where((lab.split == "cv").values & (lab.groupe.astype(str) == "1348").values
                     & (lab.silicium == False).values)[0][0]  # noqa: E712
    reel = lab[(lab.source == "Southampton_WS2_2018") & (lab.temperature_K == 295)]
    mes_i = reel.index[len(reel) // 2]
    w = (grid >= 150) & (grid <= 650)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    series = [
        ("Théorique (Materials Cloud, DFT)", g1, norm(theo), BLEU, 2.45),
        ("Simulé (dataset v2, recalé sur la mesure)", grid[w], norm(X[sim_i][w], grid[w]), VERT_EAU, 1.25),
        (f"Mesuré (Southampton, 295 K, laser {lab.excitation_nm[mes_i]:.0f} nm)", grid[w],
         norm(X[mes_i][w], grid[w]), ORANGE, 0.0),
    ]
    for nom, gx, y, c, dec in series:
        m = (gx >= 150) & (gx <= 650)
        ax.plot(gx[m], y[m] + dec, color=c, label=nom)
    for x, yy, t in [(339.6, 3.48, "E′ 340"), (402.7, 2.55, "A₁′ 403")]:
        ax.annotate(t, (x, yy), ha="center", fontsize=9, color=BLEU)
    i354 = np.argmin(abs(series[2][1] - 354)); i417 = np.argmin(abs(series[2][1] - 418))
    i520 = np.argmin(abs(series[2][1] - 520))
    for i, t in [(i354, "354"), (i417, "417"), (i520, "Si 520")]:
        ax.annotate(t, (series[2][1][i], series[2][2][i] + 0.04), ha="center", fontsize=9, color=ORANGE)
    ax.set_xlabel(AXE_X_RAMAN)
    ax.set_ylabel(AXE_Y_RAMAN + ", décalé verticalement")
    ax.set_xlim(150, 650)
    ax.set_ylim(-0.05, 3.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, fontsize=9.5)
    ax.set_title("WS₂ monocouche : le calcul place les pics 14 cm⁻¹ trop bas, la mesure ajoute un fond fort")
    return sauver(fig, racine / "figures/fig03_ws2_theorique_simule_mesure")


def fig09(racine, lab, X, grid):
    cv = lab[(lab.split == "cv")]
    choix = [("carbone", "577"), ("h-BN", "1"), ("TMD", "1348")]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5))
    for ax, (f, gid) in zip(axes, choix):
        i = cv[(cv.groupe.astype(str) == gid)].index[3]
        ax.plot(grid, norm(X[i], grid), color=COULEURS[f], lw=1)
        r = lab.loc[i]
        ax.set_title(f"{f} — {r.materiau} (id {gid}) · fond {r.fond} · Si {'oui' if r.silicium else 'non'} · "
                     f"SNR {r.snr:.0f}", fontsize=11)
        ax.set_xlabel(AXE_X_RAMAN)
        ax.set_ylabel("Intensité norm. (u.a.)")
        ax.set_xlim(100, 3200)
    fig.suptitle("Dataset v2 : un spectre simulé « comme mesuré » par famille", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return sauver(fig, racine / "figures/fig09_exemples_simules")


def fig10(racine, lab, X, grid):
    re = lab[lab.split == "test_reel"]
    choix = [re[(re.source == "Southampton_WS2_2018") & (re.temperature_K == 295)].index[10],
             re[(re.source == "Southampton_WS2_2018") & (re.temperature_K == 4)].index[10],
             re[re.source == "Southampton_HS20_2021"].index[40]]
    noms = ["WS₂ monocouche", "WS₂ monocouche", "Hétérobicouche MoSe₂/WSe₂ (HS20)"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5))
    for ax, i, nom in zip(axes, choix, noms):
        r = lab.loc[i]
        ok = np.isfinite(X[i])
        ax.plot(grid[ok], norm(X[i][ok], grid[ok]), color=ORANGE, lw=1)
        ax.set_title(f"{nom} · {r.temperature_K:.0f} K · laser {r.excitation_nm:.0f} nm", fontsize=11)
        ax.set_xlabel(AXE_X_RAMAN)
        ax.set_ylabel("Intensité norm. (u.a.)")
        ax.set_xlim(100, 660)
    fig.suptitle("Spectres mesurés du test réel : fond de photoluminescence fort sur le WS₂ brut",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return sauver(fig, racine / "figures/fig10_exemples_mesures")


def main(racine):
    racine = Path(racine)
    appliquer_style()
    d = np.load(racine / "data/v3/spectres_v3.npz")
    X, grid = d["X"], d["grid"]
    lab = pd.read_csv(racine / "data/v3/etiquettes_v3.csv")
    for f in (fig01(racine, lab), fig03(racine, lab, X, grid), fig09(racine, lab, X, grid),
              fig10(racine, lab, X, grid)):
        print(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--racine", default=".")
    main(p.parse_args().racine)
