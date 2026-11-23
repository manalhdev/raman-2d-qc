"""
Comparaison des modèles sur le dataset v3, avec trois façons d'évaluer :

  1. validation ALÉATOIRE (5 plis stratifiés)       -> montre la fuite par matériau (score trompeur)
  2. validation GROUPÉE par matériau (3 plis du v3) -> le vrai score sur un matériau jamais vu
  3. test RÉEL : entraîné sur tout le simulé, testé sur les 528 spectres mesurés (524 TMD, 4 graphite)

Deux prétraitements :
  - "recommande"   : scripts/pretraitement.py (5e centile soustrait, normalisation sans Si, fenêtres)
  - "ligne_base"   : en plus, ligne de base SNIP retirée spectre par spectre (fond de photoluminescence)

Sorties : resultats/comparaison_modeles.csv, resultats/predictions_groupees_<modele>.csv,
          resultats/predictions_reel_<modele>.csv, resultats/modele_demo.joblib

Usage : python scripts/comparer_modeles.py --racine .
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from pretraitement import masque_fenetres, normaliser_sans_si  # noqa: E402

FAMILLES = ["carbone", "h-BN", "TMD"]


def snip(y, iterations=40):
    """Ligne de base SNIP (Ryan et al. 1988) sur un spectre sans NaN. Renvoie la ligne de base."""
    v = np.log(np.log(np.sqrt(np.clip(y, 0, None) + 1) + 1) + 1)
    n = len(v)
    for k in range(1, iterations + 1):
        a = np.empty_like(v)
        a[:k] = v[:k]
        a[n - k:] = v[n - k:]
        a[k:n - k] = np.minimum(v[k:n - k], 0.5 * (v[:n - 2 * k] + v[2 * k:]))
        v = a
    return (np.exp(np.exp(v) - 1) - 1) ** 2 - 1


def retirer_ligne_base(X):
    """SNIP sur la partie mesurée de chaque spectre (les NaN restent NaN)."""
    out = np.array(X, dtype=np.float64, copy=True)
    for i, x in enumerate(out):
        ok = np.isfinite(x)
        if ok.sum() < 50:
            continue
        y = x[ok] - np.min(x[ok])
        out[i, ok] = y - snip(y)
    return out


def preparer(X, grid, ligne_base):
    if ligne_base:
        X = retirer_ligne_base(X)
    Xn = normaliser_sans_si(X, grid)[:, masque_fenetres(grid)]
    Xn = np.nan_to_num(Xn, nan=0.0)
    # moyenne par blocs de 4 cm-1 : 4x moins de variables, même information (pics larges de 4-20 cm-1)
    n = Xn.shape[1] // 4 * 4
    return Xn[:, :n].reshape(len(Xn), -1, 4).mean(axis=2).astype(np.float32)


def modeles():
    return {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=0)),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=0.5, n_jobs=-1, random_state=0, tree_method="hist"),
        "Réseau de neurones (MLP)": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, early_stopping=True, random_state=0)),
    }


def main(racine, rapide=False):
    racine = Path(racine)
    out = racine / "resultats"
    out.mkdir(exist_ok=True)
    d = np.load(racine / "data/v3/spectres_v3.npz")
    X_all, grid = d["X"], d["grid"]
    lab = pd.read_csv(racine / "data/v3/etiquettes_v3.csv")
    cv_sel = (lab.split == "cv").values
    re_sel = (lab.split == "test_reel").values
    lab_cv, lab_re = lab[cv_sel].reset_index(drop=True), lab[re_sel].reset_index(drop=True)
    enc = LabelEncoder().fit(FAMILLES)
    y = enc.transform(lab_cv.famille)
    y_re = enc.transform(lab_re.famille)
    i_tmd = list(enc.classes_).index("TMD")

    lignes = []
    for prep in ["recommande", "ligne_base"]:
        t0 = time.time()
        Xc = preparer(X_all[cv_sel], grid, prep == "ligne_base")
        Xr = preparer(X_all[re_sel], grid, prep == "ligne_base")
        print(f"[{prep}] prétraitement {time.time() - t0:.0f} s, {Xc.shape[1]} variables", flush=True)
        for nom, m in modeles().items():
            if rapide and nom not in ("Random Forest",):
                continue
            t0 = time.time()
            p_alea = cross_val_predict(m, Xc, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1)
            p_grp = cross_val_predict(m, Xc, y, cv=PredefinedSplit(lab_cv.fold.values), n_jobs=1)
            m.fit(Xc, y)
            proba_re = m.predict_proba(Xr)
            p_re = proba_re.argmax(1)
            ligne = dict(pretraitement=prep, modele=nom,
                         acc_aleatoire=(p_alea == y).mean(), acc_groupee=(p_grp == y).mean(),
                         f1_macro_aleatoire=f1_score(y, p_alea, average="macro"),
                         f1_macro_groupee=f1_score(y, p_grp, average="macro"))
            for f in FAMILLES:
                k = enc.transform([f])[0]
                ligne[f"groupee_{f}"] = (p_grp[y == k] == k).mean()
            ligne["acc_reel"] = (p_re == y_re).mean()
            for f in FAMILLES:  # rappel par famille sur le réel (vide pour h-BN : aucun spectre mesuré)
                k = enc.transform([f])[0]
                if (y_re == k).any():
                    ligne[f"reel_rappel_{f}"] = (p_re[y_re == k] == k).mean()
            for src in lab_re.source.unique():
                s = (lab_re.source == src).values
                # bonne réponse = la vraie famille de la source (plus seulement TMD depuis l'ajout du graphite)
                ligne[f"reel_{src.replace('Southampton_', '').replace('_20', '_')}"] = (p_re[s] == y_re[s]).mean()
            ligne["temps_s"] = round(time.time() - t0)
            lignes.append(ligne)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ligne.items()}, flush=True)
            cle = nom.split(" ")[0].lower()
            pd.DataFrame(dict(groupe=lab_cv.groupe, materiau=lab_cv.materiau, fold=lab_cv.fold, vrai=lab_cv.famille,
                              predit=enc.inverse_transform(p_grp))
                         ).to_csv(out / f"predictions_groupees_{prep}_{cle}.csv", index=False)
            pd.DataFrame(dict(source=lab_re.source, temperature_K=lab_re.temperature_K,
                              excitation_nm=lab_re.excitation_nm, predit=enc.inverse_transform(p_re),
                              proba_TMD=proba_re[:, i_tmd].round(3))
                         ).to_csv(out / f"predictions_reel_{prep}_{cle}.csv", index=False)
        pd.DataFrame(lignes).to_csv(out / "comparaison_modeles.csv", index=False)

    df = pd.DataFrame(lignes)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--racine", default=".")
    p.add_argument("--rapide", action="store_true")
    a = p.parse_args()
    main(a.racine, a.rapide)
