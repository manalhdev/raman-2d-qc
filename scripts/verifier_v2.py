"""Vérifications du dataset v2 : positions recalées, figure d'exemples, classifieur de référence.
Usage : python verifier_v2.py --dataset dataset_v2/ --out figures/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

BLUE, INK, INK2, GRID_C, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(Path(a.dataset) / "spectres_v2.npz")
    X, grid = d["X"], d["grid"]
    lab = pd.read_csv(Path(a.dataset) / "etiquettes_v2.csv")

    # 1. positions des pics recalés (pic principal, dans une fenêtre autour de la cible)
    for uid, lo, hi, cible in [(577, 1540, 1620, 1580), (1, 1330, 1400, 1367), (1348, 400, 440, 418)]:
        idx = np.flatnonzero(lab.id.values == uid)
        w = (grid >= lo) & (grid <= hi)
        pos = grid[w][np.argmax(X[idx][:, w], axis=1)]
        print(f"id {uid} : pic dans [{lo},{hi}] -> médiane {np.median(pos):.1f} (cible {cible}), n={len(idx)}")

    # 2. figure : un exemple par famille
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), facecolor=SURF, sharex=True)
    rng = np.random.default_rng(1)
    for ax, fam in zip(axes, ["carbone", "h-BN", "TMD"]):
        i = rng.choice(np.flatnonzero((lab.famille == fam) & (lab.split == "train") & ((lab.id == 577) | (fam != "carbone"))))
        r = lab.iloc[i]
        ax.set_facecolor(SURF)
        ax.plot(grid, X[i], color=BLUE, lw=1.2)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis="y", color=GRID_C, lw=0.8)
        ax.set_title(f"{fam} : {r.formule} (id {r.id}) · fond {r.fond} · Si {'oui' if r.silicium else 'non'} · SNR {r.snr:.0f}",
                     color=INK, fontsize=10, loc="left")
        ax.set_ylabel("Coups (simulés)", color=INK2, fontsize=9)
    axes[-1].set_xlabel("Décalage Raman (cm⁻¹)", color=INK2, fontsize=9)
    fig.suptitle("Exemples du dataset v2 : spectres simulés « comme mesurés »", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(a.out) / "exemples_v2.png", dpi=150, facecolor=SURF)

    # 3. classifieur de référence (vérification que le dataset est apprenable, pas un résultat)
    Xn = X - X.min(axis=1, keepdims=True)
    Xn = Xn / Xn.max(axis=1, keepdims=True)
    tr, te = lab.split.values == "train", lab.split.values == "test"
    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xn[tr], lab.famille[tr])
    pred = clf.predict(Xn[te])
    print(classification_report(lab.famille[te], pred, digits=3))
    print(pd.DataFrame(confusion_matrix(lab.famille[te], pred, labels=clf.classes_),
                       index=clf.classes_, columns=clf.classes_))


if __name__ == "__main__":
    main()
