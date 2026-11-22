"""
Prétraitement et chargement recommandés pour les modèles (décisions d'équipe du 18/11/2026,
finalisé le 22/11/2026 : correction de ligne de base ALS + débruitage, cf. to-do item 2).

- Fenêtres utiles : 100–1800 et 2500–2850 cm-1 seulement. Hors de ces zones, nos familles n'ont pas de pic
  (le 2D du graphène est vers 2700) : les garder n'ajoute que du bruit.
- Normalisation SANS le pic du silicium : le pic à 520 cm-1 vient du substrat, pas de l'échantillon.
  S'il sert de maximum, l'intensité du matériau dépend de l'épaisseur de silicium visible. On ignore donc
  505–535 cm-1 pour choisir le maximum.
- Ligne de base : correction ALS (Asymmetric Least Squares, Eilers & Boelens 2005) — remplace la soustraction
  de percentile utilisée jusqu'ici, qui ne suivait pas la forme réelle du fond de fluorescence.
- Débruitage : lissage Savitzky-Golay (fenêtre 11 points, ordre 3) après la ligne de base.
- Pas d'imputation ni de SMOTE : le dataset simulé n'a ni NaN, ni doublon, et ses classes sont équilibrées.
  Moyenner deux spectres (SMOTE) créerait des pics doublés. Les NaN des spectres réels = hors plage mesurée :
  ils sont mis à 0 APRÈS normalisation (pas de signal mesuré à cet endroit). Pendant la correction de ligne de
  base et le débruitage, les NaN sont interpolés temporairement (sinon le solveur ALS ne les accepte pas),
  puis restaurés en NaN avant la normalisation.
- Validation : toujours groupée par matériau (colonne `fold` ou `groupe`), jamais aléatoire.

Exemple :
    from pretraitement import charger
    X, y, groupes, plis, lab, grille = charger("data/v3", jeu="cv")      # simulé, pour la validation croisée
    Xr, yr, *_ = charger("data/v3", jeu="test_reel")                   # mesures réelles

Note de performance : la correction ALS résout un système creux par spectre et par itération (10 itérations
par défaut). Sur les ~4250 spectres de data/v3, compter de l'ordre de quelques minutes pour `charger(...)`.
Rien n'empêche de mettre les résultats en cache (ex. np.save) si ça devient gênant en boucle d'entraînement.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

FENETRES = [(100, 1800), (2500, 2850)]
ZONE_SILICIUM = (505, 535)


def masque_fenetres(grid):
    m = np.zeros_like(grid, dtype=bool)
    for lo, hi in FENETRES:
        m |= (grid >= lo) & (grid <= hi)
    return m


def als_baseline(y, lam=1e5, p=0.01, niter=10, D=None):
    """Ligne de base d'un seul spectre par Asymmetric Least Squares (Eilers & Boelens, 2005).

    lam : lissage de la ligne de base (plus grand = plus lisse, suit moins les pics).
    p : asymétrie (0 < p < 1) ; petit p pénalise peu les points sous la ligne de base, qui suit donc
        le creux du spectre (fluorescence) sans suivre les pics Raman, bien plus étroits.
    D : matrice de pénalité précalculée (lam * D2.D2^T), pour éviter de la reconstruire à chaque appel
        quand on traite beaucoup de spectres de même longueur (voir `corriger_ligne_base`).
    """
    y = np.asarray(y, dtype=np.float64)
    L = len(y)
    if D is None:
        D2 = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=np.float64)
        D = lam * D2.dot(D2.transpose())
    w = np.ones(L)
    z = y
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def corriger_ligne_base(X, grid, lam=1e5, p=0.01, niter=10):
    """Applique `als_baseline` à chaque ligne (spectre) de X, en se limitant à sa plage réellement
    mesurée (les NaN marquent le hors-plage pour les spectres réels). Travailler sur la sous-plage
    plutôt que d'interpoler sur toute la grille évite un artefact de bord (marche artificielle que
    l'ALS essaie de suivre, provoquant un faux rebond juste avant/après la plage mesurée)."""
    X = np.asarray(X, dtype=np.float64)
    n, L = X.shape
    cache_D = {}

    def D_pour(longueur):
        if longueur not in cache_D:
            D2 = sparse.diags([1, -2, 1], [0, -1, -2], shape=(longueur, longueur - 2), dtype=np.float64)
            cache_D[longueur] = lam * D2.dot(D2.transpose())
        return cache_D[longueur]

    out = np.full_like(X, np.nan)
    for i in range(n):
        y = X[i]
        ok = np.flatnonzero(np.isfinite(y))
        if ok.size < 5:
            continue
        lo, hi = ok[0], ok[-1] + 1  # plage mesurée, supposée contiguë
        sous_y = y[lo:hi]
        if np.any(~np.isfinite(sous_y)):  # trous isolés dans la plage mesurée : comblés par interpolation
            idx = np.arange(sous_y.size)
            bons = np.isfinite(sous_y)
            sous_y = np.interp(idx, idx[bons], sous_y[bons])
        base = als_baseline(sous_y, lam=lam, p=p, niter=niter, D=D_pour(hi - lo))
        out[i, lo:hi] = sous_y - base
    return out


def debruiter(X, fenetre=11, ordre=3):
    """Lissage Savitzky-Golay le long de chaque spectre (axe des colonnes), limité à la plage
    mesurée de chaque spectre (NaN conservés) — même raison qu'en ligne de base : lisser à travers
    un bord NaN/0 créerait un artefact."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    out = np.full_like(X, np.nan)
    for i in range(n):
        y = X[i]
        ok = np.flatnonzero(np.isfinite(y))
        if ok.size < fenetre:
            out[i, ok] = y[ok]
            continue
        lo, hi = ok[0], ok[-1] + 1
        out[i, lo:hi] = savgol_filter(y[lo:hi], fenetre, ordre, mode="interp")
    return out


def normaliser_sans_si(X, grid, lam=1e5, p=0.01, niter=10, fenetre=11, ordre=3):
    """Prétraitement final : correction de ligne de base ALS, débruitage Savitzky-Golay, puis
    normalisation par le maximum hors zone du silicium. NaN conservés (= hors plage mesurée)."""
    Xb = corriger_ligne_base(X, grid, lam=lam, p=p, niter=niter)
    Xd = debruiter(Xb, fenetre=fenetre, ordre=ordre)
    hors_si = ~((grid >= ZONE_SILICIUM[0]) & (grid <= ZONE_SILICIUM[1]))
    m = np.nanmax(np.where(hors_si, Xd, np.nan), axis=1, keepdims=True)
    m[~np.isfinite(m) | (m <= 0)] = 1.0
    return (Xd / m).astype(np.float32)


def preparer(X, grid):
    """Normalisation sans silicium (ALS + débruitage inclus), puis fenêtres utiles, puis NaN -> 0."""
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
