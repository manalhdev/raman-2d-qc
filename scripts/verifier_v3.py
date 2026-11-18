"""Vérifications du dataset v3 : figure de spectres réels et test du modèle de référence (entraîné sur le
synthétique v2) sur les spectres mesurés.
Usage : python verifier_v3.py --v2 dataset_v2/ --v3 dataset_v3/ --out figures/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BLUE, INK, INK2, GRID_C, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def norm(X):
    X = np.nan_to_num(X, nan=0.0)
    X = X - X.min(axis=1, keepdims=True)
    return X / (X.max(axis=1, keepdims=True) + 1e-12)


def pretraiter(X, grid):
    """Prétraitement MINIMAL (démonstration, pas la méthode du projet) :
    trous comblés par interpolation, ligne de base = minimum glissant lissé (fenêtre 151 points), soustraite."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    out = np.empty_like(X, dtype=np.float32)
    for i, y in enumerate(X):
        ok = np.isfinite(y)
        y = np.interp(grid, grid[ok], y[ok]) if ok.any() else np.zeros_like(grid)
        base = uniform_filter1d(minimum_filter1d(y, 151, mode="nearest"), 151, mode="nearest")
        out[i] = np.clip(y - base, 0, None)
    return norm(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d2 = np.load(Path(a.v2) / "spectres_v2.npz"); X2, grid = d2["X"], d2["grid"]
    l2 = pd.read_csv(Path(a.v2) / "etiquettes_v2.csv")
    d3 = np.load(Path(a.v3) / "spectres_reels.npz"); Xr = d3["X"]
    lr = pd.read_csv(Path(a.v3) / "etiquettes_reels.csv")

    # figure : 3 spectres réels représentatifs (zone 100-700 cm-1)
    choix = [("Southampton_WS2_2018", 295), ("Southampton_WS2_2018", 4), ("Southampton_HS20_2021", 4)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), facecolor=SURF, sharex=True)
    for ax, (src, T) in zip(axes, choix):
        idx = np.flatnonzero((lr.source == src) & (lr.temperature_K == T))
        i = idx[len(idx) // 2]
        r = lr.iloc[i]
        m = (grid <= 700)
        ax.set_facecolor(SURF)
        ax.plot(grid[m], Xr[i][m], color=BLUE, lw=1.2)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis="y", color=GRID_C, lw=0.8)
        ax.set_title(f"{r.materiau} ({r.structure}) · {T} K · laser {r.excitation_nm:.0f} nm", color=INK, fontsize=10, loc="left")
        ax.set_ylabel("Intensité (unités de la source)", color=INK2, fontsize=8)
    axes[-1].set_xlabel("Décalage Raman (cm⁻¹)", color=INK2, fontsize=9)
    fig.suptitle("Spectres Raman MESURÉS ajoutés dans le dataset v3 (Southampton, CC BY 4.0)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(a.out) / "spectres_reels_v3.png", dpi=150, facecolor=SURF)

    # test : modèle de référence entraîné sur le synthétique (v2), testé sur le réel
    # même plage que la mesure : on coupe le synthétique à 100-650 cm-1 pour une comparaison juste
    w = (grid >= 100) & (grid <= 620)
    tr = np.ones(len(l2), bool)  # entraînement sur tout le simulé ; le test est le réel
    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    clf.fit(norm(X2[tr][:, w]), l2.famille[tr])
    acc_synth = float("nan")  # pas de test simulé ici : voir verifier_v2 (validation groupée)
    pred = clf.predict(norm(Xr[:, w]))
    lr["pred_reference"] = pred
    print("[Sans prétraitement] plage 100-620 cm-1, entraîné sur tout le simulé")
    print("Spectres réels prédits TMD (bonne réponse) :")
    print(lr.groupby("source").pred_reference.apply(lambda s: f"{(s == 'TMD').mean():.1%} ({(s == 'TMD').sum()}/{len(s)})").to_string())

    g = grid[w]
    clf2 = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    clf2.fit(pretraiter(X2[tr][:, w], g), l2.famille[tr])
    acc2 = float("nan")
    lr["pred_pretraite"] = clf2.predict(pretraiter(Xr[:, w], g))
    print("[Avec ligne de base]")
    print(lr.groupby("source").pred_pretraite.apply(lambda s: f"{(s == 'TMD').mean():.1%} ({(s == 'TMD').sum()}/{len(s)})").to_string())
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pretraitement import normaliser_sans_si
    wr = (grid >= 100) & (grid <= 620)
    def prep3(X):
        return np.nan_to_num(pretraiter(normaliser_sans_si(X, grid)[:, wr], grid[wr]), nan=0.0)
    clf3 = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(prep3(X2), l2.famille)
    lr["pred_recommande"] = clf3.predict(prep3(Xr))
    print("[Ligne de base + normalisation sans silicium]")
    print(lr.groupby("source").pred_recommande.apply(lambda s: f"{(s == 'TMD').mean():.1%} ({(s == 'TMD').sum()}/{len(s)})").to_string())
    lr[["source", "temperature_K", "excitation_nm", "pred_reference", "pred_pretraite", "pred_recommande"]].to_csv(Path(a.v3) / "test_modele_reference_sur_reel.csv", index=False)


if __name__ == "__main__":
    main()
