"""Protocoles d'évaluation.

- « aléatoire » : StratifiedKFold(3) mélangé. Les mêmes matériaux sont au train et au test : mesure surtout la
  mémorisation. Donné seulement pour comparaison.
- « groupé »   : PredefinedSplit sur la colonne `fold` du v3 (3 plis, chaque matériau dans un seul pli).
  Le modèle est testé sur des matériaux qu'il n'a jamais vus : c'est le chiffre à retenir.
- « réel »     : entraînement sur tout le simulé, test sur les 521 spectres mesurés (tous TMD).
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import PredefinedSplit, StratifiedKFold

from .donnees import FAMILLES
from .modeles import SEED


def plis_groupes(folds):
    ps = PredefinedSplit(np.asarray(folds))
    s = list(ps.split())
    return s


def plis_aleatoires(y, n=3, seed=SEED):
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def predictions_hors_pli(fabrique, X, y, plis, proba=False):
    """Prédictions out-of-fold : chaque spectre est prédit par un modèle qui ne l'a pas vu."""
    out = np.zeros((len(y), len(FAMILLES))) if proba else np.zeros(len(y), dtype=int)
    for a, b in plis:
        m = fabrique().fit(X[a], y[a])
        out[b] = m.predict_proba(X[b]) if proba else m.predict(X[b])
    return out


def scores(y, pred, prefixe=""):
    r = {f"{prefixe}accuracy": accuracy_score(y, pred), f"{prefixe}f1_macro": f1_score(y, pred, average="macro")}
    for k, c in enumerate(FAMILLES):
        if (y == k).any():
            r[f"{prefixe}acc_{c}"] = accuracy_score(y[y == k], pred[y == k])
    return r


def comparer(nom, fabrique, X, y, folds):
    """Évalue un modèle en validation aléatoire ET groupée. Retourne (ligne de résultats, préd. groupées)."""
    t = time.time()
    p_alea = predictions_hors_pli(fabrique, X, y, plis_aleatoires(y))
    p_grp = predictions_hors_pli(fabrique, X, y, plis_groupes(folds))
    r = {"modele": nom, **scores(y, p_alea, "aleatoire_"), **scores(y, p_grp, "groupe_"),
         "temps_s": round(time.time() - t, 1)}
    return r, p_grp


def taux_tmd_par_source(pred, sources):
    """Sur le réel (tous TMD), part des spectres classés TMD, par source."""
    df = pd.DataFrame({"source": sources, "ok": np.asarray(pred) == FAMILLES.index("TMD")})
    t = df.groupby("source").ok.mean()
    t["TOTAL"] = df.ok.mean()
    return t
