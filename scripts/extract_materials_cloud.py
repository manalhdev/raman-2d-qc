"""
Extraction du jeu Materials Cloud Archive 2024.148 (Li et al., 2024)
et construction du dataset de spectres Raman pour le PI² n°156.

Source : https://archive.materialscloud.org/record/2024.148
         DOI 10.24435/materialscloud:bq-vp, licence CC BY 4.0
Contenu de data.zip : data/<id>_<formule>/{geometry.in, POSCAR, raman.dat}
raman.dat : colonne 1 = fréquence (cm-1), colonne 2 = intensité Raman (Å^4/amu)

Ce que fait le script :
 1. lit les 3504 matériaux (structure + liste des modes Raman calculés) ;
 2. attribue une famille : carbone, h-BN, TMD, ou autre ;
 3. transforme chaque liste de raies en spectre continu (raies lorentziennes)
    sur la grille commune 100–3200 cm-1, pas de 1 cm-1, puis normalise (max = 1).

Usage :
    python extract_materials_cloud.py --zip data.zip --out dataset/ [--fwhm 8]
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import spglib

# ----------------------------------------------------------------------------
# Paramètres par défaut (décidés par l'équipe le 17/11/2026)
# ----------------------------------------------------------------------------
GRID_MIN, GRID_MAX, GRID_STEP = 100.0, 3200.0, 1.0
DEFAULT_FWHM = 8.0  # largeur à mi-hauteur des raies lorentziennes (cm-1), hypothèse

TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}
CHALCOGENS = {"S", "Se", "Te"}


# ----------------------------------------------------------------------------
# Lecture des fichiers
# ----------------------------------------------------------------------------
def parse_poscar(text):
    lines = text.strip().splitlines()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    coord_type = lines[7].strip().lower()
    n = sum(counts)
    pos = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(n)])
    if coord_type.startswith("c") or coord_type.startswith("k"):
        pos = pos @ np.linalg.inv(lattice)
    numbers = []
    for k, c in enumerate(counts):
        numbers += [k + 1] * c
    return lattice, pos, numbers, dict(zip(species, counts)), species


def parse_raman(text):
    arr = np.loadtxt(io.StringIO(text), ndmin=2)
    if arr.size == 0:
        return np.zeros((0, 2))
    return arr[:, :2]


# ----------------------------------------------------------------------------
# Famille du matériau
# ----------------------------------------------------------------------------
def famille(composition):
    els = set(composition)
    if els == {"C"}:
        return "carbone"
    if els == {"B", "N"} and composition["B"] == composition["N"]:
        return "h-BN"
    metals = els & TRANSITION_METALS
    chalc = els & CHALCOGENS
    if len(metals) == 1 and chalc and els == metals | chalc:
        m = composition[next(iter(metals))]
        x = sum(composition[c] for c in chalc)
        if x == 2 * m:  # stœchiométrie MX2 (inclut les Janus MXY)
            return "TMD"
    return "autre"


# ----------------------------------------------------------------------------
# Spectre continu
# ----------------------------------------------------------------------------
def lorentz_spectrum(freqs, intens, grid, fwhm):
    g = fwhm / 2.0
    y = np.zeros_like(grid)
    for f, a in zip(freqs, intens):
        y += a * g**2 / ((grid - f) ** 2 + g**2)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fwhm", type=float, default=DEFAULT_FWHM)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = np.arange(GRID_MIN, GRID_MAX + GRID_STEP / 2, GRID_STEP)

    zf = zipfile.ZipFile(args.zip)
    folders = sorted({n.split("/")[1] for n in zf.namelist() if n.count("/") >= 2 and n.split("/")[1]},
                     key=lambda s: int(s.split("_")[0]))

    meta, modes_rows, spectra = [], [], []
    for folder in folders:
        uid, formula = folder.split("_", 1)
        poscar = zf.read(f"data/{folder}/POSCAR").decode()
        lattice, pos, numbers, comp, species = parse_poscar(poscar)
        try:  # 3 dossiers de l'archive n'ont pas de raman.dat
            modes = parse_raman(zf.read(f"data/{folder}/raman.dat").decode())
            raman_present = True
        except KeyError:
            modes, raman_present = np.zeros((0, 2)), False

        try:
            ds = spglib.get_symmetry_dataset((lattice, pos, numbers), symprec=1e-2)
            sg_symbol, sg_num = ds.international, ds.number
        except Exception:
            sg_symbol, sg_num = "", 0

        z_cart = (pos @ lattice)[:, 2]
        thickness = float(z_cart.max() - z_cart.min())

        active = modes[(modes[:, 0] > 1.0) & (modes[:, 1] > 1e-6)] if len(modes) else modes
        for f, a in modes:
            modes_rows.append((int(uid), formula, round(float(f), 4), round(float(a), 6)))

        y = lorentz_spectrum(active[:, 0], active[:, 1], grid, args.fwhm) if len(active) else np.zeros_like(grid)
        ymax = y.max()
        spectra.append((y / ymax).astype(np.float32) if ymax > 0 else y.astype(np.float32))

        main_peak = float(active[np.argmax(active[:, 1]), 0]) if len(active) else np.nan
        meta.append(dict(
            id=int(uid), formule=formula, famille=famille(comp),
            elements="-".join(species), n_atomes=len(numbers),
            groupe_espace=sg_symbol, num_groupe_espace=sg_num,
            epaisseur_A=round(thickness, 3),
            n_modes=len(modes), n_modes_actifs=len(active),
            pic_principal_cm1=round(main_peak, 2),
            n_modes_actifs_dans_grille=int(((active[:, 0] >= GRID_MIN) & (active[:, 0] <= GRID_MAX)).sum()) if len(active) else 0,
            raman_dat_present=raman_present,
            spectre_vide=bool(ymax <= 0),
        ))

    meta = pd.DataFrame(meta)
    X = np.vstack(spectra)
    cols = [f"{int(w)}" for w in grid]

    meta.to_csv(out / "metadata_tous.csv", index=False)
    pd.DataFrame(modes_rows, columns=["id", "formule", "frequence_cm1", "intensite_A4_amu"]).to_csv(
        out / "modes_raman_tous.csv", index=False)
    np.savez_compressed(out / "spectres_tous.npz", X=X, grid=grid, id=meta.id.values,
                        formule=meta.formule.values, famille=meta.famille.values, fwhm=args.fwhm)

    keep = meta.famille.isin(["carbone", "h-BN", "TMD"]) & ~meta.spectre_vide
    sub = pd.concat([meta.loc[keep, ["id", "formule", "famille", "groupe_espace", "n_atomes"]].reset_index(drop=True),
                     pd.DataFrame(X[keep.values], columns=cols)], axis=1)
    sub.to_csv(out / "spectres_3familles.csv", index=False, float_format="%.5f")

    print("Matériaux lus :", len(meta))
    print(meta.famille.value_counts().to_string())
    print("Spectres vides (aucun mode actif) :", int(meta.spectre_vide.sum()))
    print("Dataset 3 familles :", int(keep.sum()), "spectres x", len(grid), "points")


if __name__ == "__main__":
    main()
