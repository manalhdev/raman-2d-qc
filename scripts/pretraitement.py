"""
Prétraitement et chargement recommandés pour les modèles (décisions d'équipe du 18/11/2026).

- Fenêtres utiles : 100–1800 et 2500–2850 cm-1 seulement. Hors de ces zones, nos familles n'ont pas de pic
  (le 2D du graphène est vers 2700) : les garder n'ajoute que du bruit.
- Normalisation SANS le pic du silicium : le pic à 520 cm-1 vient du substrat, pas de l'échantillon.
  S'il sert de maximum, l'intensité du matériau dépend de l'épaisseur de silicium visible. On ignore donc
  505–535 cm-1 pour choisir le maximum.
- Ligne de base : 5e centile du spectre soustrait (simple ; un vrai prétraitement de ligne de base reste l'étape 1).
- Pas d'imputation ni de SMOTE : le dataset simulé n'a ni NaN, ni doublon, et ses classes sont équilibrées.
  Moyenner deux spectres (SMOTE) créerait des pics doublés. Les NaN des spectres réels = hors plage mesurée :
  ils sont mis à 0 APRÈS normalisation (pas de signal mesuré à cet endroit).
- Validation : toujours groupée par matériau (colonne `fold` ou `groupe`), jamais aléatoire.

Exemple :
    from pretraitement import charger
    X, y, groupes, plis, lab, grille = charger("data/v3", jeu="cv")      # simulé, pour la validation croisée
    Xr, yr, *_ = charger("data/v3", jeu="test_reel")                   # mesures réelles
"""
from pathlib import Path

import numpy as np
import pandas as pd

FENETRES = [(100, 1800), (2500, 2850)]
ZONE_SILICIUM = (505, 535)


def masque_fenetres(grid):
    m = np.zeros_like(grid, dtype=bool)
    for lo, hi in FENETRES:
        m |= (grid >= lo) & (grid <= hi)
    return m


def normaliser_sans_si(X, grid):
    """Soustrait le 5e centile puis divise par le maximum hors zone du silicium. NaN conservés."""
    X = np.asarray(X, dtype=np.float32)
    base = np.nanpercentile(X, 5, axis=1, keepdims=True)
    Xc = X - base
    hors_si = ~((grid >= ZONE_SILICIUM[0]) & (grid <= ZONE_SILICIUM[1]))
    m = np.nanmax(np.where(hors_si, Xc, np.nan), axis=1, keepdims=True)
    m[~np.isfinite(m) | (m <= 0)] = 1.0
    return Xc / m


def preparer(X, grid):
    """Normalisation sans silicium, puis fenêtres utiles, puis NaN -> 0."""
    Xn = normaliser_sans_si(X, grid)[:, masque_fenetres(grid)]
    return np.nan_to_num(Xn, nan=0.0), grid[masque_fenetres(grid)]


def charger(dossier, jeu="cv"):
    """Charge le dataset v3 (ou v2) prétraité. jeu = "cv" (simulé, plis groupés) ou "test_reel"."""
    dossier = Path(dossier)
    fichier = "spectres_v3.npz" if (dossier / "spectres_v3.npz").exists() else "spectres_v2.npz"
    etiq = "etiquettes_v3.csv" if fichier == "spectres_v3.npz" else "etiquettes_v2.csv"
    d = np.load(dossier / fichier)
    lab = pd.read_csv(dossier / etiq)
    if "split" in lab.columns:
        sel = (lab.split == jeu).values
    else:
        sel = np.ones(len(lab), bool) if jeu == "cv" else np.zeros(len(lab), bool)
    X, grille = preparer(d["X"][sel], d["grid"])
    lab = lab[sel].reset_index(drop=True)
    plis = lab["fold"].values if "fold" in lab.columns else None
    return X, lab.famille.values, lab.groupe.values if "groupe" in lab.columns else None, plis, lab, grille
