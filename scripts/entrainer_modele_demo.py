"""
Entraîne le modèle de la démo sur TOUT le simulé (v3, split cv) et prépare des spectres d'exemple.

Sorties :
  demo/modele_demo.joblib      {"modele", "pretraitement", "familles", "grille", "infos"}
  demo/exemples/*.csv          spectres d'exemple, 2 colonnes : nombre_onde_cm1, intensite

Usage : python scripts/entrainer_modele_demo.py --racine . [--modele "Random Forest"] [--pretraitement recommande]
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comparer_modeles import FAMILLES, modeles, preparer  # noqa: E402


def main(racine, nom_modele, prep):
    racine = Path(racine)
    d = np.load(racine / "data/v3/spectres_v3.npz")
    X, grid = d["X"], d["grid"]
    lab = pd.read_csv(racine / "data/v3/etiquettes_v3.csv")
    cv = (lab.split == "cv").values
    m = modeles()[nom_modele]
    Xc = preparer(X[cv], grid, prep == "ligne_base")
    y = lab.famille[cv].values
    m.fit(Xc, y)
    (racine / "demo/exemples").mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(modele=m, pretraitement=prep, familles=list(m.classes_), grille=grid,
                     infos=f"{nom_modele}, prétraitement {prep}, entraîné sur {cv.sum()} spectres simulés"),
                racine / "demo/modele_demo.joblib", compress=3)

    def exporte(i, nom):
        ok = np.isfinite(X[i])
        pd.DataFrame(dict(nombre_onde_cm1=grid[ok], intensite=X[i][ok].round(4))).to_csv(
            racine / f"demo/exemples/{nom}.csv", index=False)

    re = lab[lab.split == "test_reel"]
    exporte(re[(re.source == "Southampton_WS2_2018") & (re.temperature_K == 295)].index[10], "mesure_WS2_295K")
    exporte(re[(re.source == "Southampton_WS2_2018") & (re.temperature_K == 4)].index[3], "mesure_WS2_4K")
    exporte(re[re.source == "Southampton_HS20_2021"].index[40], "mesure_MoSe2_WSe2_HS20")
    s = lab[cv]
    exporte(s[s.groupe.astype(str) == "577"].index[3], "simule_graphene")
    exporte(s[s.groupe.astype(str) == "1"].index[3], "simule_hBN")
    print("modèle et exemples écrits dans demo/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--racine", default=".")
    p.add_argument("--modele", default="Random Forest")
    p.add_argument("--pretraitement", default="recommande")
    a = p.parse_args()
    main(a.racine, a.modele, a.pretraitement)
