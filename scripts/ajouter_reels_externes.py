"""
Ajoute au v3 les spectres Raman MESURÉS hors Southampton : 6 spectres RRUFF (Meriam, 3 graphite,
2 MoS2, 1 WS2) et le graphite Mendeley (Pierre-Louis). Tous à 532 nm, massifs, plage large :
ce sont les premiers spectres réels de carbone et les premiers qui couvrent 100-1800 cm-1.

Le script ajoute ces spectres à data/v3/spectres_reels.npz et etiquettes_reels.csv (en retirant
d'abord les lignes de ces sources s'il a déjà tourné : on peut le relancer sans doublon), puis il
faut relancer assembler_v3.py.

    python scripts/ajouter_reels_externes.py --sources data/sources/ --v3 data/v3/
    python scripts/assembler_v3.py --v2 data/v2/ --v3 data/v3/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GRID = np.arange(100.0, 3200.5, 1.0)

# (fichier, source, famille, matériau, structure)
SPECTRES = [
    ("rruff/R050503_graphite_532nm_raw.txt", "RRUFF_R050503", "carbone", "graphite", "massif"),
    ("rruff/R090047_graphite_532nm_raw.txt", "RRUFF_R090047", "carbone", "graphite", "massif"),
    ("rruff/R120025_graphite_532nm_raw.txt", "RRUFF_R120025", "carbone", "graphite", "massif (avec défauts, identification non confirmée)"),
    ("rruff/R050209_MoS2_532nm_raw.txt", "RRUFF_R050209", "TMD", "MoS2", "massif 2H"),
    ("rruff/R060124_MoS2_532nm_raw.txt", "RRUFF_R060124", "TMD", "MoS2", "massif"),
    ("rruff/R070616_WS2_532nm_raw.txt", "RRUFF_R070616", "TMD", "WS2", "massif 3R (mélangé avec PbS)"),
    ("mendeley_czdwt595yp_graphite_raman.txt", "Mendeley_graphite_2019", "carbone", "graphite", "massif (poudre)"),
]
PREFIXES = ("RRUFF_", "Mendeley_")


def lire(fichier):
    """Fichier RRUFF (en-têtes ##, 'x, y') ou 2 colonnes séparées par des espaces."""
    xs, ys = [], []
    for ligne in Path(fichier).read_text(encoding="utf-8", errors="ignore").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#"):
            continue
        v = ligne.replace(",", " ").split()
        try:
            xs.append(float(v[0])); ys.append(float(v[1]))
        except (ValueError, IndexError):
            continue
    x, y = np.array(xs), np.array(ys)
    o = np.argsort(x)
    return x[o], y[o]


def sur_grille(x, y):
    """Interpolation linéaire sur la grille du v3 (1 cm-1). NaN hors plage mesurée."""
    g = np.full(GRID.shape, np.nan, dtype=np.float32)
    m = (GRID >= x[0]) & (GRID <= x[-1])
    g[m] = np.interp(GRID[m], x, y)
    return g


def main(sources, v3):
    sources, v3 = Path(sources), Path(v3)
    d = np.load(v3 / "spectres_reels.npz")
    lab = pd.read_csv(v3 / "etiquettes_reels.csv")
    garde = ~lab.source.str.startswith(PREFIXES).values
    X, lab = d["X"][garde], lab[garde].reset_index(drop=True)
    nouveaux, lignes = [], []
    for fichier, source, famille, materiau, structure in SPECTRES:
        x, y = lire(sources / fichier)
        nouveaux.append(sur_grille(x, y))
        lignes.append(dict(famille=famille, source=source, materiau=materiau, structure=structure,
                           temperature_K=293, excitation_nm=532.0, excitation_eV=round(1239.84 / 532.0, 4),
                           cm1_min=round(float(x[0]), 1), cm1_max=round(float(x[-1]), 1),
                           traite_par_auteurs=False))
        print(f"{source:24s} {famille:8s} {materiau:9s} {x[0]:7.1f}-{x[-1]:7.1f} cm-1")
    X = np.vstack([X, np.array(nouveaux, dtype=X.dtype)])
    lab = pd.concat([lab, pd.DataFrame(lignes)], ignore_index=True)
    np.savez_compressed(v3 / "spectres_reels.npz", X=X, grid=d["grid"])
    lab.to_csv(v3 / "etiquettes_reels.csv", index=False)
    print(f"spectres_reels : {len(lab)} spectres ({lab.famille.value_counts().to_dict()})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="data/sources/")
    p.add_argument("--v3", default="data/v3/")
    a = p.parse_args()
    main(a.sources, a.v3)
