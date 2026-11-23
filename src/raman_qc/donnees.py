"""Chargement du dataset v3 (spectres bruts) et classement des sources de données."""
from pathlib import Path

import numpy as np
import pandas as pd

FAMILLES = ["carbone", "h-BN", "TMD"]
COULEURS = {"carbone": "#2a6f97", "h-BN": "#c8553d", "TMD": "#588157"}


def trouver_racine(depart=None):
    """Remonte depuis `depart` (ou le dossier courant) jusqu'au dossier qui contient data/v3."""
    p = Path(depart or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / "data" / "v3" / "spectres_v3.npz").exists():
            return d
    raise FileNotFoundError("data/v3/spectres_v3.npz introuvable : lancer depuis le dépôt raman-2d-qc")


def charger_v3(racine=None):
    """Spectres BRUTS du v3 (NaN = hors plage mesurée), grille en cm-1 et étiquettes.

    Retourne X (4271 x 3101, float32), grid (3101,), lab (DataFrame, colonne `y` = indice de famille).
    """
    racine = trouver_racine(racine)
    d = np.load(racine / "data" / "v3" / "spectres_v3.npz")
    lab = pd.read_csv(racine / "data" / "v3" / "etiquettes_v3.csv")
    lab["y"] = lab.famille.map({c: i for i, c in enumerate(FAMILLES)}).astype(int)
    return d["X"].astype(np.float32), d["grid"].astype(np.float64), lab


def catalogue_sources(racine=None):
    """Classe toutes les données du dépôt en théorique / expérimental (retour au tuteur du 19/11).

    Une ligne par source, avec ce qu'elle contient et l'usage qu'on en fait.
    """
    racine = trouver_racine(racine)
    _, _, lab = charger_v3(racine)
    n = lab.groupby("source").size().to_dict()
    zen = pd.read_csv(racine / "data" / "v1" / "zenodo_MoS2_mesures.csv")
    lignes = [
        dict(categorie="Théorique - calculé (DFT)", source="Materials Cloud 2024.148 (v1)",
             contenu="Modes Raman calculés par DFT, élargis en spectres", familles="carbone, h-BN, TMD",
             n=133, usage="Spectres de référence (base des simulations)"),
        dict(categorie="Théorique - simulé", source="Materials_Cloud_2024.148_augmente (v2/v3, split cv)",
             contenu="Spectres DFT + largeur, décalage, bruit, fond, pic Si", familles="carbone, h-BN, TMD",
             n=n.get("Materials_Cloud_2024.148_augmente", 0), usage="Entraînement et validation croisée groupée"),
        dict(categorie="Expérimental - brut", source="Southampton WS2 2018 (DOI 10.5258/SOTON/D0371)",
             contenu="Raman de résonance, 4-295 K, fond de photoluminescence", familles="TMD (WS2)",
             n=n.get("Southampton_WS2_2018", 0), usage="Test sur réel (le plus difficile)"),
        dict(categorie="Expérimental - traité par les auteurs",
             source="Southampton HS20 + HS57 2021 (DOI 10.5258/SOTON/D1747)",
             contenu="Hétérobicouche MoSe2/WSe2, 4 K, fond déjà soustrait", familles="TMD (MoSe2/WSe2)",
             n=n.get("Southampton_HS20_2021", 0) + n.get("Southampton_HS57_2021", 0), usage="Test sur réel"),
        dict(categorie="Expérimental - brut, 532 nm, plage large", source="RRUFF (R050503, R090047, R120025, R050209, R060124, R070616)",
             contenu="Minéraux massifs : 3 graphite, 2 MoS2, 1 WS2 ; 86-6630 cm-1", familles="carbone, TMD",
             n=sum(v for k, v in n.items() if k.startswith("RRUFF")), usage="Test sur réel, plage complète"),
        dict(categorie="Expérimental - brut, 532 nm, plage large", source="Mendeley graphite (DOI 10.17632/czdwt595yp.1)",
             contenu="Graphite en poudre, 20 °C, 502-3499 cm-1", familles="carbone",
             n=n.get("Mendeley_graphite_2019", 0), usage="Test sur réel, plage complète"),
        dict(categorie="Expérimental - positions de pics seulement", source="Zenodo MoS2 (v1)",
             contenu="Positions E2g / A1g, contrainte, dopage (pas de spectre)", familles="TMD (MoS2)",
             n=len(zen), usage="Écart théorie / mesure sur les positions de pics"),
    ]
    return pd.DataFrame(lignes)
