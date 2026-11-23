"""Prétraitement des spectres (étape 1 du sujet).

Chaîne : découpe de la plage utile -> lissage Savitzky-Golay -> ligne de base -> normalisation sans le pic
du silicium -> NaN (hors plage mesurée) mis à 0.

Ligne de base ALS (Asymmetric Least Squares, Eilers & Boelens 2005) : on cherche une courbe lisse z qui passe
SOUS le spectre. On minimise  sum w_i (y_i - z_i)^2 + lam * sum (Δ² z_i)^2, avec un poids w_i = p si le point est
au-dessus de z (un pic) et 1 - p sinon. Peu de paramètres, robuste aux fonds de photoluminescence.
"""
import numpy as np
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter

FENETRES_COMPLETES = [(100, 1800), (2500, 2850)]   # décision d'équipe du 18/11 (scripts/pretraitement.py)
PLAGE_COMMUNE_REEL = [(100, 620)]                   # plage mesurée par (presque) tous les spectres réels
ZONE_SILICIUM = (505, 535)


def masque(grid, fenetres):
    m = np.zeros_like(grid, dtype=bool)
    for lo, hi in fenetres:
        m |= (grid >= lo) & (grid <= hi)
    return m


def _als_1d(y, lam, p, n_iter):
    """ALS sur un spectre sans NaN. Système pentadiagonal symétrique résolu en O(n)."""
    n = len(y)
    # D'D pour la différence seconde, stocké en bandes (format 'upper' de solveh_banded)
    d0 = np.full(n, 6.0); d0[[0, -1]] = 1; d0[[1, -2]] = 5
    d1 = np.full(n - 1, -4.0); d1[[0, -1]] = -2
    d2 = np.ones(n - 2)
    w = np.ones(n)
    for _ in range(n_iter):
        ab = np.zeros((3, n))
        ab[0, 2:] = lam * d2
        ab[1, 1:] = lam * d1
        ab[2] = lam * d0 + w
        z = solveh_banded(ab, w * y)
        w_new = np.where(y > z, p, 1 - p)
        if np.array_equal(w_new, w):
            break
        w = w_new
    return z


def ligne_de_base_als(X, lam=1e5, p=0.01, n_iter=10):
    """Ligne de base ALS ligne par ligne. Les NaN (hors plage mesurée) sont ignorés puis conservés."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    B = np.full_like(X, np.nan)
    for i, y in enumerate(X):
        ok = np.isfinite(y)
        if ok.sum() > 10:
            B[i, ok] = _als_1d(y[ok], lam, p, n_iter)
    return B


def ligne_de_base_centile(X, q=5):
    """Version simple de l'équipe (18/11) : un seul niveau = 5e centile du spectre."""
    return np.broadcast_to(np.nanpercentile(X, q, axis=1, keepdims=True), X.shape)


def _lisser(X, fenetre=7, ordre=2):
    out = X.copy()
    for i, y in enumerate(X):
        ok = np.isfinite(y)
        if ok.sum() > fenetre:
            out[i, ok] = savgol_filter(y[ok], fenetre, ordre)
    return out


def normaliser_sans_si(X, grid):
    """Divise par le maximum hors de la zone du silicium (505-535 cm-1)."""
    hors_si = ~((grid >= ZONE_SILICIUM[0]) & (grid <= ZONE_SILICIUM[1]))
    m = np.nanmax(np.where(hors_si, X, np.nan), axis=1, keepdims=True)
    m[~np.isfinite(m) | (m <= 0)] = 1.0
    return X / m


def pretraiter(X, grid, fenetres=FENETRES_COMPLETES, ligne_de_base="als", lissage=True,
               lam=1e5, p=0.01):
    """Chaîne complète. ligne_de_base = "als" | "centile" | "aucune".

    Retourne (Xp, grille) : spectres prétraités sans NaN, sur la grille des fenêtres choisies.
    """
    m = masque(grid, fenetres)
    Xw, g = np.asarray(X, dtype=np.float64)[:, m], grid[m]
    if lissage:
        Xw = _lisser(Xw)
    if ligne_de_base == "als":
        # ALS fenêtre par fenêtre : pas de ligne de base continue à travers un trou de la grille
        for lo, hi in fenetres:
            k = (g >= lo) & (g <= hi)
            Xw[:, k] = Xw[:, k] - ligne_de_base_als(Xw[:, k], lam=lam, p=p)
        Xw = np.clip(Xw, 0, None)
    elif ligne_de_base == "centile":
        Xw = Xw - ligne_de_base_centile(Xw)
    elif ligne_de_base != "aucune":
        raise ValueError(ligne_de_base)
    if ligne_de_base == "aucune":
        Xw = Xw - np.nanmin(Xw, axis=1, keepdims=True)   # simple décalage à 0 pour pouvoir normaliser
    Xw = np.clip(normaliser_sans_si(Xw, g), None, 5.0)   # un pic Si géant ne doit pas écraser le reste
    return np.nan_to_num(Xw, nan=0.0).astype(np.float32), g
