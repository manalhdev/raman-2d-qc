"""Caractéristiques extraites des spectres prétraités (étape 2 du sujet)."""
import numpy as np
from scipy.signal import find_peaks

from .pretraitement import ZONE_SILICIUM


def spectre_par_bandes(P, largeur=4):
    """Moyenne par bandes de `largeur` points (1 point = 1 cm-1) : réduit le bruit et la taille d'entrée."""
    n = P.shape[1] // largeur * largeur
    return P[:, :n].reshape(len(P), -1, largeur).mean(2)


def noms_pics(n=8):
    return [f"position_{i}" for i in range(n)] + [f"hauteur_{i}" for i in range(n)] + [f"largeur_{i}" for i in range(n)]


def pics(P, grid, n=8, prominence=0.05):
    """Les n pics les plus proéminents (hors silicium) : position (cm-1), hauteur, largeur (cm-1).

    Triés par position ; -1 si moins de n pics. 3n valeurs physiquement interprétables.
    """
    pas = float(np.median(np.diff(grid)))
    out = np.full((len(P), 3 * n), -1.0, dtype=np.float32)
    for i, s in enumerate(P):
        pk, pr = find_peaks(s, prominence=prominence, width=1)
        hors_si = (grid[pk] < ZONE_SILICIUM[0]) | (grid[pk] > ZONE_SILICIUM[1])
        pk, h, w = pk[hors_si], pr["prominences"][hors_si], pr["widths"][hors_si]
        o = np.argsort(h)[::-1][:n]
        o = o[np.argsort(pk[o])]
        k = len(o)
        out[i, :k] = grid[pk[o]]
        out[i, n:n + k] = h[o]
        out[i, 2 * n:2 * n + k] = w[o] * pas
    return out
