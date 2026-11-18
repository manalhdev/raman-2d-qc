"""
Dataset v2 : recalage + augmentation réaliste + plis de validation GROUPÉS PAR MATÉRIAU.

Part des raies calculées (modes_raman_tous.csv) des 133 matériaux des 3 familles et produit
des spectres « comme mesurés » : pics de largeur variable, décalage de calibration, bruit,
fond de fluorescence, pic du silicium, intensités variables. Classes équilibrées.

Usage :
    python augmenter_dataset.py --dataset dataset/ --out dataset_v2/ [--n-par-famille 1250 --n-plis 3 --seed 0]

Correction du 18/11/2026 : l'ancienne version mettait les MÊMES matériaux de carbone et de h-BN (3 chacun)
en entraînement et en test (fuite par matériau : 97 % en validation aléatoire, 36 % en validation groupée).
Désormais chaque spectre porte son matériau (`groupe`) et un pli (`fold`) : tous les spectres d'un matériau
sont dans le même pli. Évaluer avec ces plis (GroupKFold) ; ne jamais mélanger les spectres d'un matériau.

Tous les paramètres de simulation sont des CHOIX PAR DÉFAUT, à valider en équipe (voir PARAMS).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID = np.arange(100.0, 3200.5, 1.0)  # grille commune décidée le 17/11/2026
FAMILLES = ["carbone", "h-BN", "TMD"]

# ----------------------------------------------------------------------------
# 1. Recalage sur des positions mesurées (sources dans le README)
#    (id, [(fréquence calculée, fréquence mesurée), ...])
# ----------------------------------------------------------------------------
RECALAGE = {
    577: [(1443.1, 1580.0)],                 # graphène, G (Ferrari & Basko 2013)
    1: [(1356.3, 1367.0)],                   # h-BN monocouche, E2g (1366 massif, +1 à +4 en monocouche : Gorbachev et al. 2011)
    1348: [(339.7, 356.0), (402.7, 418.0)],  # WS2 monocouche, E' et A1' (~356 / ~417-418 cm-1)
}
FENETRE_ANCRE = 15.0  # un mode à moins de 15 cm-1 d'une ancre reçoit le décalage de cette ancre

# ----------------------------------------------------------------------------
# 2. Paramètres de simulation (hypothèses, à valider)
# ----------------------------------------------------------------------------
PARAMS = dict(
    fwhm_min=4.0, fwhm_max=20.0,          # largeur à mi-hauteur des pics (cm-1)
    fwhm_jitter=0.2,                      # chaque pic : largeur x [0.8, 1.2]
    decalage_global=10.0,                 # calibration + erreur résiduelle du calcul : ±10 cm-1
    decalage_pic=3.0,                     # chaque pic : ±3 cm-1 en plus (contrainte, dopage)
    sigma_intensite=0.3,                  # intensité relative de chaque pic x lognormal(0, 0.3)
    p_silicium=0.5, si_pos=520.5, si_fwhm=(3.5, 6.0), si_rel=(0.1, 2.0),
    p_fond=0.8, fond_rel=(0.05, 1.0),     # fond de fluorescence, relatif au pic principal
    snr=(15.0, 300.0),                    # rapport signal sur bruit (pic principal / écart-type du bruit)
    comptes_max=(500.0, 20000.0),         # hauteur du pic principal en « coups »
    # graphène : bande 2D absente du calcul (processus du 2nd ordre), ajoutée d'après la littérature
    g2d_pos=(2670.0, 2700.0), g2d_fwhm=(24.0, 35.0), g2d_ratio=(1.0, 4.0),
)


def recaler(uid, freqs):
    if uid not in RECALAGE:
        return freqs
    ancres = RECALAGE[uid]
    ratio_moyen = np.mean([e / c for c, e in ancres])
    out = freqs.copy()
    for i, f in enumerate(freqs):
        d = [abs(f - c) for c, _ in ancres]
        k = int(np.argmin(d))
        out[i] = f + (ancres[k][1] - ancres[k][0]) if d[k] <= FENETRE_ANCRE else f * ratio_moyen
    return out


def lorentz(grid, pos, fwhm, amp):
    g = fwhm / 2.0
    return amp * g**2 / ((grid - pos) ** 2 + g**2)


def fond(rng, grid):
    t = (grid - grid[0]) / (grid[-1] - grid[0])
    kind = rng.integers(3)
    if kind == 0:  # polynôme lent
        c = rng.uniform(-1, 1, 4)
        y = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3
        y = y - y.min()
    elif kind == 1:  # fluorescence montante
        y = np.exp(rng.uniform(0.5, 3.0) * t) - 1
    else:  # bosse large
        y = np.exp(-0.5 * ((t - rng.uniform(0, 1)) / rng.uniform(0.15, 0.6)) ** 2)
    return y / (y.max() + 1e-12), ["polynome", "exponentiel", "bosse"][kind]


def simuler(rng, uid, famille, freqs, amps):
    p = PARAMS
    fw = rng.uniform(p["fwhm_min"], p["fwhm_max"])
    shift = rng.uniform(-p["decalage_global"], p["decalage_global"])
    y = np.zeros_like(GRID)
    for f, a in zip(freqs, amps):
        y += lorentz(GRID,
                     f + shift + rng.uniform(-p["decalage_pic"], p["decalage_pic"]),
                     fw * rng.uniform(1 - p["fwhm_jitter"], 1 + p["fwhm_jitter"]),
                     a * rng.lognormal(0, p["sigma_intensite"]))
    if uid == 577:  # graphène : bande 2D
        g = freqs[np.argmax(amps)]
        y += lorentz(GRID, rng.uniform(*p["g2d_pos"]) + shift, rng.uniform(*p["g2d_fwhm"]),
                     y.max() * rng.uniform(*p["g2d_ratio"]))
    y = y / y.max()
    si = rng.random() < p["p_silicium"]
    if si:
        y += lorentz(GRID, p["si_pos"] + shift, rng.uniform(*p["si_fwhm"]), rng.uniform(*p["si_rel"]))
    has_fond = rng.random() < p["p_fond"]
    ftype = "aucun"
    if has_fond:
        b, ftype = fond(rng, GRID)
        y += rng.uniform(*p["fond_rel"]) * b
    snr = np.exp(rng.uniform(np.log(p["snr"][0]), np.log(p["snr"][1])))
    y += rng.normal(0, 1.0 / snr, size=y.shape)
    y *= rng.uniform(*p["comptes_max"])
    return y.astype(np.float32), dict(fwhm=round(fw, 2), decalage=round(shift, 2), silicium=si,
                                      fond=ftype, snr=round(snr, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-par-famille", type=int, default=1250, help="spectres simulés par famille")
    ap.add_argument("--n-plis", type=int, default=3, help="plis de validation groupée (3 = un matériau de carbone et de h-BN par pli)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    src, out = Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(src / "metadata_tous.csv")
    base = meta[meta.famille.isin(FAMILLES) & ~meta.spectre_vide].copy()
    modes = pd.read_csv(src / "modes_raman_tous.csv")
    modes = modes[(modes.frequence_cm1 > 1) & (modes.intensite_A4_amu > 1e-6)]

    # ---- plis de validation groupés par matériau ----
    # Chaque matériau (id Materials Cloud) va entièrement dans UN pli ; dans chaque famille les matériaux sont
    # répartis à tour de rôle entre les plis, pour que chaque pli contienne les trois familles.
    base["fold"] = -1
    for fam in FAMILLES:
        ids = base.loc[base.famille == fam, "id"].values.copy()
        rng.shuffle(ids)
        for k, uid in enumerate(ids):
            base.loc[base.id == uid, "fold"] = k % a.n_plis
    base[["id", "formule", "famille", "groupe_espace", "fold"]].to_csv(out / "plis_materiaux.csv", index=False)

    raies = {uid: (g.frequence_cm1.values, g.intensite_A4_amu.values) for uid, g in modes.groupby("id")}

    X, rows = [], []
    for fam in FAMILLES:
        ids = base[base.famille == fam].id.values
        for k in range(a.n_par_famille):
            uid = int(ids[k % len(ids)])
            f, amp = raies[uid]
            y, info = simuler(rng, uid, fam, recaler(uid, f), amp)
            X.append(y)
            r = base.loc[base.id == uid].iloc[0]
            rows.append(dict(groupe=uid, fold=int(r.fold), id=uid, formule=r.formule, famille=fam,
                             recale=uid in RECALAGE, **info))
    X = np.vstack(X)
    lab = pd.DataFrame(rows)
    np.savez_compressed(out / "spectres_v2.npz", X=X, grid=GRID)
    lab.to_csv(out / "etiquettes_v2.csv", index=False)
    (out / "parametres_simulation.json").write_text(json.dumps(
        dict(seed=a.seed, n_par_famille=a.n_par_famille, n_plis=a.n_plis, params=PARAMS,
             recalage={str(k): v for k, v in RECALAGE.items()}), indent=2, ensure_ascii=False))

    print(lab.groupby(["fold", "famille"]).agg(spectres=("id", "size"), materiaux=("id", "nunique")).to_string())
    print("X :", X.shape)


if __name__ == "__main__":
    main()
