"""
Figures « résultats » (à lancer après scripts/comparer_modeles.py).

  fig04_comparaison_modeles        aléatoire vs groupée vs test réel, pour chaque modèle
  fig05_scores_par_famille         carte des scores : famille (validation groupée) et source réelle
  fig06_predictions_carbone_hbn    que devient un matériau de carbone / h-BN jamais vu ?
  fig07_zones_utilisees            zones du spectre utilisées par la Random Forest + spectres moyens
  fig08_reel_selon_temperature     test réel WS2 : score selon la température de mesure

Usage : python scripts/figures_resultats.py --racine . [--modele "Random Forest"]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from comparer_modeles import preparer  # noqa: E402
from pretraitement import masque_fenetres  # noqa: E402
from style_figures import (AXE_X_RAMAN, BLEU, COULEURS, ENCRE, ENCRE_2, ORANGE, VERT_EAU,  # noqa: E402
                           appliquer_style, sauver)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PREP = "recommande"
NOMS_MAT = {"577": "graphène (577)", "543": "carbone Cmmm (543)", "1849": "carbone Cmme (1849)",
            "1": "h-BN hexagonal (1)", "83": "BN P2₁/c (83)", "3278": "BN P4/nmm (3278)"}


def fig04(racine, res):
    r = res[res.pretraitement == PREP].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(r))
    w = 0.26
    for j, (col, nom, c) in enumerate([("acc_aleatoire", "Validation aléatoire (fuite)", BLEU),
                                       ("acc_groupee", "Validation groupée par matériau", ORANGE),
                                       ("acc_reel", "Test sur 528 spectres mesurés (524 TMD, 4 graphite)", VERT_EAU)]):
        b = ax.bar(x + (j - 1) * w, 100 * r[col], width=w - 0.03, color=c, label=nom)
        ax.bar_label(b, fmt="%.0f", fontsize=8.5, color=ENCRE_2, padding=2)
    ax.set_xticks(x, r.modele, fontsize=9.5)
    ax.set_ylabel("Exactitude (%)")
    ax.set_ylim(0, 112)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
    lo, hi = 100 * r.acc_aleatoire.min(), 100 * r.acc_aleatoire.max()
    g_lo, g_hi = 100 * r.acc_groupee.min(), 100 * r.acc_groupee.max()
    ax.set_title(f"Aléatoire : {lo:.0f}-{hi:.0f} % ; sur un matériau jamais vu : {g_lo:.0f}-{g_hi:.0f} % "
                 "pour tous les modèles")
    return sauver(fig, racine / "figures/fig04_comparaison_modeles")


def fig05(racine, res):
    r = res.copy()
    r["nom"] = r.modele + np.where(r.pretraitement == "ligne_base", " + ligne de base", "")
    r = r.set_index("nom")
    cols = [("groupee_carbone", "carbone\n(groupée)"), ("groupee_h-BN", "h-BN\n(groupée)"),
            ("groupee_TMD", "TMD\n(groupée)"), ("reel_WS2_18", "WS₂ mesuré\n(289)"),
            ("reel_HS20_21", "MoSe₂/WSe₂\nHS20 (102)"), ("reel_HS57_21", "MoSe₂/WSe₂\nHS57 (130)")]
    M = 100 * r[[c for c, _ in cols]].values
    cmap = LinearSegmentedColormap.from_list("bleu", ["#f2f6fc", "#9cc3ef", BLEU, "#0f3f7a"])
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f} %", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] > 60 else ENCRE)
    ax.set_xticks(range(len(cols)), [n for _, n in cols], fontsize=9.5)
    ax.set_yticks(range(len(r)), r.index)
    ax.grid(False)
    ax.axvline(2.5, color="white", lw=4)
    ax.axhline(5.5, color="white", lw=4)
    ax.set_xlabel("Validation groupée : part des spectres de la famille bien classés · "
                  "Test réel : part reconnue comme TMD")
    ax.set_title("Le carbone et le h-BN ne sont presque jamais reconnus sur un matériau nouveau")
    return sauver(fig, racine / "figures/fig05_scores_par_famille")


def fig06(racine, cle, prep):
    p = pd.read_csv(racine / f"resultats/predictions_groupees_{prep}_{cle}.csv")
    p = p[p.vrai.isin(["carbone", "h-BN"])]
    p["groupe"] = p.groupe.astype(str)
    ordre = ["577", "543", "1849", "1", "83", "3278"]
    t = pd.crosstab(p.groupe, p.predit, normalize="index").reindex(ordre).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(ordre))
    gauche = np.zeros(len(ordre))
    for f in ["carbone", "h-BN", "TMD"]:
        v = t[f].values if f in t else np.zeros(len(ordre))
        ax.barh(y, v, left=gauche, color=COULEURS[f], label=f"prédit {f}", height=0.62,
                edgecolor="white", linewidth=2)
        for i in range(len(ordre)):
            if v[i] >= 8:
                ax.text(gauche[i] + v[i] / 2, y[i], f"{v[i]:.0f} %", ha="center", va="center", fontsize=9,
                        color="white")
        gauche += v
    ax.set_yticks(y, [f"{NOMS_MAT[g]} — vrai : {'carbone' if g in ('577', '543', '1849') else 'h-BN'}"
                      for g in ordre])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Part des spectres du matériau (%) — modèle {cle.upper()} ({prep}), chaque matériau testé sans avoir été vu")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
    ax.set_title("Un matériau de carbone ou de h-BN jamais vu est classé TMD ou dans l'autre famille")
    return sauver(fig, racine / "figures/fig06_predictions_carbone_hbn")


def fig07(racine):
    from sklearn.ensemble import RandomForestClassifier
    d = np.load(racine / "data/v3/spectres_v3.npz")
    X, grid = d["X"], d["grid"]
    lab = pd.read_csv(racine / "data/v3/etiquettes_v3.csv")
    cv = (lab.split == "cv").values
    Xc = preparer(X[cv], grid, False)
    g = grid[masque_fenetres(grid)]
    g = g[: len(g) // 4 * 4].reshape(-1, 4).mean(1)
    y = lab.famille[cv].values
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Xc, y)
    imp = rf.feature_importances_
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex="col",
                             gridspec_kw=dict(width_ratios=[1700, 350], height_ratios=[1, 1.3], wspace=0.04))
    for col, (lo, hi) in enumerate([(100, 1800), (2500, 2850)]):
        m = (g >= lo) & (g <= hi)
        a0, a1 = axes[0, col], axes[1, col]
        a0.fill_between(g[m], 100 * imp[m], color=ENCRE_2, lw=0, step="mid")
        for f in ["carbone", "h-BN", "TMD"]:
            a1.plot(g[m], Xc[y == f][:, m].mean(0), color=COULEURS[f], label=f, lw=1.4)
        a1.set_xlim(lo, hi)
        a1.set_xlabel(AXE_X_RAMAN if col == 0 else "")
        if col == 1:
            a0.tick_params(labelleft=False)
            a1.tick_params(labelleft=False)
            a0.set_ylim(axes[0, 0].get_ylim())
            a1.set_ylim(axes[1, 0].get_ylim())
    axes[0, 0].set_ylabel("Importance (%)")
    axes[1, 0].set_ylabel("Spectre moyen\nnormalisé (u.a.)")
    axes[1, 0].legend(loc="upper right")
    top = g[np.argsort(imp)[::-1][:5]]
    fig.suptitle("La forêt aléatoire s'appuie sur les pics de matériaux précis (BN 677 et 950, h-BN 1366, carbone 1437) :\n"
                 "elle reconnaît des matériaux déjà vus plutôt que des familles",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    return sauver(fig, racine / "figures/fig07_zones_utilisees")


def fig08(racine, cle):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for prep, nom, c in [("recommande", "Prétraitement recommandé", BLEU),
                         ("ligne_base", "+ ligne de base SNIP", ORANGE)]:
        f = racine / f"resultats/predictions_reel_{prep}_{cle}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        w = p[p.source == "Southampton_WS2_2018"]
        s = w.groupby("temperature_K").apply(lambda d: 100 * (d.predit == "TMD").mean(), include_groups=False)
        ax.plot(s.index, s.values, marker="o", ms=7, color=c, label=nom)
    ax.set_xlabel("Température de la mesure (K)")
    ax.set_ylabel("WS₂ reconnu comme TMD (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.set_title("SVM : avec la ligne de base SNIP, le WS₂ mesuré est reconnu de 4 à 295 K (97-100 %)")
    return sauver(fig, racine / "figures/fig08_reel_selon_temperature")


def main(racine, modele, prep):
    racine = Path(racine)
    appliquer_style()
    res = pd.read_csv(racine / "resultats/comparaison_modeles.csv")
    cle = modele.split(" ")[0].lower()
    for f in (fig04(racine, res), fig05(racine, res), fig06(racine, cle, prep), fig07(racine), fig08(racine, cle)):
        print(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--racine", default=".")
    p.add_argument("--modele", default="SVM (RBF)")
    p.add_argument("--pretraitement", default="ligne_base")
    a = p.parse_args()
    main(a.racine, a.modele, a.pretraitement)
