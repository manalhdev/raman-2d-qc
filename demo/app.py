"""
Démo PI² n°156 — on dépose un spectre Raman, l'outil affiche la famille prédite et sa confiance.

Lancer (depuis la racine du dépôt) :
    pip install -r requirements.txt streamlit
    python scripts/entrainer_modele_demo.py --racine .     # une fois, crée demo/modele_demo.joblib
    streamlit run demo/app.py

Format du fichier : CSV ou TXT à 2 colonnes (nombre d'onde en cm-1, intensité), avec ou sans en-tête.
"""
import io
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks, savgol_filter

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))
from comparer_modeles import preparer, snip  # noqa: E402
from style_figures import AXE_X_RAMAN, COULEURS, ORANGE, appliquer_style  # noqa: E402

REFERENCES = [  # (position cm-1, nom) : valeurs mesurées de la littérature et de nos données
    (241, "MoSe₂ A₁′"), (250, "WSe₂ A₁′/E′"), (287, "MoSe₂ E′"), (355, "WS₂ E′+2LA"), (384, "MoS₂ E¹₂g"),
    (404, "MoS₂ A₁g"), (418, "WS₂ A₁′"), (520, "Silicium (substrat)"), (1350, "Graphène D (défauts)"),
    (1367, "h-BN E₂g"), (1580, "Graphène G"), (2680, "Graphène 2D"),
]


@st.cache_resource
def charger_modele():
    return joblib.load(RACINE / "demo/modele_demo.joblib")


def lire_spectre(contenu):
    df = pd.read_csv(io.BytesIO(contenu), sep=None, engine="python", header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    x, y = df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float)
    o = np.argsort(x)
    return x[o], y[o]


def sur_grille(x, y, grille):
    v = np.full(len(grille), np.nan)
    dans = (grille >= x.min()) & (grille <= x.max())
    v[dans] = np.interp(grille[dans], x, y)
    return v


def pics(x, y):
    yy = y - y.min()
    yy = yy - snip(yy)                      # retire le fond (photoluminescence, pente)
    fen = max(9, (len(yy) // 250) | 1)      # lissage adapté au nombre de points
    if len(yy) > fen:
        yy = savgol_filter(yy, fen, 2)
    p, prop = find_peaks(yy, prominence=0.15 * yy.max(), distance=5)
    p = p[np.argsort(prop["prominences"])[::-1][:8]]
    lignes = []
    for i in sorted(p):
        ref = min(REFERENCES, key=lambda r: abs(r[0] - x[i]))
        proche = f"{ref[1]} ({ref[0]})" if abs(ref[0] - x[i]) <= 7 else "—"
        lignes.append({"Position (cm⁻¹)": round(float(x[i]), 1), "Pic de référence le plus proche (±7 cm⁻¹)": proche})
    return pd.DataFrame(lignes), p


st.set_page_config(page_title="Raman 2D — démo PI² 156", layout="wide")
st.title("Identification d'un matériau 2D à partir de son spectre Raman")
st.caption("PI² n°156 · ESILV · démo v1 (novembre 2026) — familles reconnues : carbone, h-BN, TMD")

M = charger_modele()
grille = M["grille"]
exemples = sorted((RACINE / "demo/exemples").glob("*.csv"))

gauche, droite = st.columns([1, 2])
with gauche:
    fichier = st.file_uploader("Déposer un spectre (CSV/TXT : nombre d'onde, intensité)", type=["csv", "txt"])
    choix = st.selectbox("…ou choisir un exemple", ["—"] + [e.stem for e in exemples])
    contenu = fichier.getvalue() if fichier else (
        (RACINE / f"demo/exemples/{choix}.csv").read_bytes() if choix != "—" else None)

if contenu is None:
    st.info("Déposez un fichier ou choisissez un exemple pour lancer l'analyse.")
    st.stop()

x, y = lire_spectre(contenu)
v = sur_grille(x, y, grille)
Xp = preparer(v[None, :], grille, M["pretraitement"] == "ligne_base")
proba = M["modele"].predict_proba(Xp)[0]
fam = M["familles"]
k = int(np.argmax(proba))

with gauche:
    st.subheader("Résultat")
    if proba[k] >= 0.6:
        st.success(f"Famille prédite : **{fam[k]}** — confiance {proba[k]:.0%}")
    else:
        st.warning(f"Résultat incertain : **{fam[k]}** à {proba[k]:.0%} seulement. "
                   "Le spectre ressemble peu aux données d'entraînement.")
    st.bar_chart(pd.DataFrame({"Probabilité": proba}, index=fam))
    st.caption(f"Plage mesurée : {x.min():.0f} à {x.max():.0f} cm⁻¹ · {len(x)} points · modèle : {M['infos']}")

with droite:
    appliquer_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    yn = (y - np.percentile(y, 1)) / (np.max(y - np.percentile(y, 1)) or 1)
    ax.plot(x, yn, color=COULEURS.get(fam[k], ORANGE), lw=1.2)
    tab, p = pics(x, y)
    ax.scatter(x[p], yn[p], color="#0b0b0b", s=18, zorder=3)
    for i in p:
        ax.annotate(f"{x[i]:.0f}", (x[i], yn[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel(AXE_X_RAMAN)
    ax.set_ylabel("Intensité normalisée (u.a.)")
    ax.set_title("Spectre déposé et pics détectés")
    st.pyplot(fig)
    st.dataframe(tab, hide_index=True)

st.divider()
st.caption("Limites connues : carbone et h-BN appris sur 3 matériaux chacun (score faible sur un matériau nouveau) ; "
           "pas encore d'indicateur de qualité ; spectres de résonance à basse température plus difficiles.")
