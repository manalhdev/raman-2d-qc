"""Entraînement des modèles finaux et analyse d'un nouveau spectre (base de la démo).

    python -m raman_qc.pipeline entrainer            # depuis src/ ou avec PYTHONPATH=src
    python -m raman_qc.pipeline analyser mon_spectre.csv   # 2 colonnes : nombre d'onde (cm-1), intensité

Deux jeux de modèles sont entraînés sur TOUT le simulé :
- « complet » : plage 100-1800 + 2500-2850 cm-1 (spectre de routine qui couvre G, 2D, h-BN)
- « court »   : plage 100-620 cm-1 (comme les mesures Southampton) ; ne distingue pas carbone et h-BN
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from .donnees import FAMILLES, charger_v3, trouver_racine
from .features import pics, spectre_par_bandes
from sklearn.neighbors import NearestNeighbors

from .modeles import Conforme, xgboost
from .pretraitement import FENETRES_COMPLETES, PLAGE_COMMUNE_REEL, pretraiter

PLAGES = {"complet": FENETRES_COMPLETES, "court": PLAGE_COMMUNE_REEL}
BANDES = {"complet": 4, "court": 2}


def _features(P, g, plage):
    """Plage complète : features de pics (seules à généraliser à un matériau nouveau, cf. notebook).
    Plage courte : spectre par bandes après ALS (meilleur score sur les mesures réelles)."""
    return pics(P, g) if plage == "complet" else spectre_par_bandes(P, BANDES[plage])


def entrainer(dossier_sortie=None, racine=None):
    racine = trouver_racine(racine)
    sortie = Path(dossier_sortie or racine / "modeles")
    sortie.mkdir(exist_ok=True)
    X, grid, lab = charger_v3(racine)
    cv = (lab.split == "cv").values
    y, folds = lab.y.values[cv], lab.fold.values[cv]
    info = {}
    for plage, fen in PLAGES.items():
        P, g = pretraiter(X[cv], grid, fen, ligne_de_base="als")
        F = _features(P, g, plage)
        # calibration conforme sur un pli de matériaux jamais vus (plus honnête qu'un tirage aléatoire)
        a, c = folds != 2, folds == 2
        m_cal = xgboost().fit(F[a], y[a])
        conf = Conforme(m_cal, alpha=0.05).calibrer(F[c], y[c])
        m = xgboost().fit(F, y)
        # hors domaine : distance cosinus au plus proche spectre d'entraînement (AUC 0,96-1,0 dans le notebook,
        # contre 0,63-0,83 pour l'Isolation Forest). Seuil = 99e centile des distances entre spectres connus.
        B = spectre_par_bandes(P, BANDES[plage]).astype(np.float32)
        nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(B)
        seuil = float(np.quantile(nn.kneighbors(B)[0][:, 1], 0.99))
        joblib.dump({"modele": m, "q_conforme": float(conf.q_), "voisins": nn, "seuil_distance": seuil,
                     "plage": plage}, sortie / f"xgb_{plage}.joblib", compress=3)
        info[plage] = {"q_conforme": float(conf.q_), "seuil_distance": seuil, "n_features": F.shape[1]}
    (sortie / "info.json").write_text(json.dumps(info, indent=2))
    return info


def analyser(nombres_onde, intensites, dossier_modeles=None, plage=None):
    """Classe un spectre mesuré. Retourne famille prédite, probabilités, ensemble conforme 95 % et alerte."""
    nombres_onde, intensites = np.asarray(nombres_onde, float), np.asarray(intensites, float)
    o = np.argsort(nombres_onde)
    grid = np.arange(100, 3201, dtype=float)
    x = np.interp(grid, nombres_onde[o], intensites[o], left=np.nan, right=np.nan)[None]
    if plage is None:
        plage = "complet" if nombres_onde.max() >= 1700 else "court"
    dossier = Path(dossier_modeles or trouver_racine() / "modeles")
    M = joblib.load(dossier / f"xgb_{plage}.joblib")
    P, g = pretraiter(x, grid, PLAGES[plage], ligne_de_base="als")
    proba = M["modele"].predict_proba(_features(P, g, plage))[0]
    ensemble = [FAMILLES[k] for k in np.where(1 - proba <= M["q_conforme"])[0]]
    distance = float(M["voisins"].kneighbors(spectre_par_bandes(P, BANDES[plage]))[0][0, 0])
    r = {"plage_modele": plage, "famille": FAMILLES[int(proba.argmax())],
         "probabilites": {f: round(float(v), 3) for f, v in zip(FAMILLES, proba)}, "ensemble_95": ensemble,
         "hors_domaine": distance > M["seuil_distance"], "distance_plus_proche": round(distance, 3),
         "spectre_pretraite": (g, P[0])}
    if plage == "court":
        r["attention"] = "spectre < 1700 cm-1 : carbone et h-BN n'ont pas de pic dans cette plage, seul « TMD ou non » est fiable"
    return r


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "entrainer":
        print(entrainer())
    elif len(sys.argv) > 2 and sys.argv[1] == "analyser":
        import pandas as pd
        d = pd.read_csv(sys.argv[2], sep=None, engine="python", comment="#", header=None)
        d = d.apply(pd.to_numeric, errors="coerce").dropna().values   # ignore une éventuelle ligne d'en-tête
        r = analyser(d[:, 0], d[:, 1])
        r.pop("spectre_pretraite")
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print(__doc__)
