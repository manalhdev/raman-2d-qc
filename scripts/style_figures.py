"""
Style commun de toutes les figures du projet (rapport et présentation).

Règles (retour de l'encadrante, 22/11/2026) : chaque figure a un titre, des axes nommés AVEC unité,
une légende dès qu'il y a 2 séries, et elle est exportée en PNG 300 dpi + SVG.

    from style_figures import appliquer_style, COULEURS, sauver
    appliquer_style()
    fig, ax = plt.subplots()
    ...
    sauver(fig, "figures/fig01_exemple")
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Palette catégorielle validée (ordre fixe, jamais recyclé)
BLEU, ORANGE, VERT_EAU, JAUNE, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
ENCRE, ENCRE_2, GRILLE, FOND = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

# Une couleur par famille, la même dans TOUTES les figures
COULEURS = {"carbone": BLEU, "h-BN": ORANGE, "TMD": VERT_EAU}
# Une couleur par type de donnée
TYPE = {"théorique": BLEU, "mesuré": ORANGE}

AXE_X_RAMAN = "Nombre d'onde (cm⁻¹)"
AXE_Y_RAMAN = "Intensité normalisée (u.a.)"


def appliquer_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "figure.facecolor": FOND, "axes.facecolor": FOND,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 11, "axes.labelcolor": ENCRE, "axes.edgecolor": ENCRE_2, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.6,
        "xtick.color": ENCRE_2, "ytick.color": ENCRE_2, "text.color": ENCRE,
        "legend.frameon": False, "legend.fontsize": 10, "lines.linewidth": 1.6,
    })


def sauver(fig, chemin_sans_ext):
    chemin = Path(chemin_sans_ext)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(chemin.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return chemin.with_suffix(".png")
