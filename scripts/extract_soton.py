"""
Extraction des spectres Raman MESURÉS de l'Université de Southampton (licence CC BY 4.0).

Sources :
 - McDonnell et al. (2018), WS2 monocouche, Raman de résonance de 4 K à 295 K.
   Dataset DOI 10.5258/SOTON/D0371 — https://eprints.soton.ac.uk/416765 — fichier RRS_Spectra.xlsx
   Feuilles « <T> Shift », « <T> Intensity », « <T> WL » (une colonne = un spectre).
 - Viner, McDonnell, Smith (2021), hétérostructures MoSe2/WSe2 encapsulées dans du hBN, 4 K.
   Dataset DOI 10.5258/SOTON/D1747 — https://eprints.soton.ac.uk/449632 — HS20_Data.xlsx, HS57_Data.xlsx
   Feuille « Processed Data » : pour chaque spectre, énergie d'excitation puis 2 colonnes (décalage, intensité).
   Données déjà traitées par les auteurs (fond soustrait, intensité calibrée).

Chaque spectre est interpolé sur la grille commune 100–3200 cm-1 (pas de 1). Les points hors de la plage
mesurée valent NaN (pas de donnée) : les colonnes cm1_min / cm1_max donnent la plage couverte.

Usage :
    python extract_soton.py --ws2 RRS_Spectra.xlsx --hs20 HS20_Data.xlsx --hs57 HS57_Data.xlsx --out dataset_v3/
"""
import argparse
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

GRID = np.arange(100.0, 3200.5, 1.0)
HC_EV_NM = 1239.84193  # énergie (eV) = 1239.84 / longueur d'onde (nm)


def sur_grille(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    o = np.argsort(x)
    x, y = x[o], y[o]
    # bords du détecteur : intensité < 2 % de la médiane au début et à la fin (coupure du filtre) -> pas une mesure
    seuil = 0.02 * np.median(np.abs(y[len(y) // 4: 3 * len(y) // 4])) if len(y) > 8 else 0
    ok = np.abs(y) > seuil
    if ok.any():
        i0, i1 = np.argmax(ok), len(ok) - np.argmax(ok[::-1])
        x, y = x[i0:i1], y[i0:i1]
    g = np.full(GRID.shape, np.nan, dtype=np.float32)
    m = (GRID >= x[0]) & (GRID <= x[-1])
    g[m] = np.interp(GRID[m], x, y)
    # trous dans la mesure (écart > 3 pas) : pas d'interpolation à travers
    pas = np.median(np.diff(x))
    for k in np.flatnonzero(np.diff(x) > 3 * pas):
        g[(GRID > x[k]) & (GRID < x[k + 1])] = np.nan
    return g, float(x[0]), float(x[-1])


def lire_ws2(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    noms = {ws.title.strip(): ws.title for ws in wb.worksheets}
    temps = sorted({t.split()[0] for t in noms}, key=lambda s: int(s.rstrip("K")))
    out = []
    for T in temps:
        S = np.array(list(wb[noms[f"{T} Shift"]].iter_rows(values_only=True)), dtype=float)
        I = np.array(list(wb[noms[f"{T} Intensity"]].iter_rows(values_only=True)), dtype=float)
        WL = list(wb[noms[f"{T} WL"]].iter_rows(values_only=True))[0]
        for j in range(S.shape[1]):
            y, lo, hi = sur_grille(S[:, j], I[:, j])
            out.append((y, dict(source="Southampton_WS2_2018", materiau="WS2", structure="monocouche",
                                temperature_K=int(T.rstrip("K")), excitation_nm=round(float(WL[j]), 2),
                                excitation_eV=round(HC_EV_NM / float(WL[j]), 4),
                                cm1_min=round(lo, 1), cm1_max=round(hi, 1), traite_par_auteurs=False)))
    return out


def lire_hs(path, nom):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb["Processed Data"].iter_rows(values_only=True))
    hdr = rows[0]
    out = []
    for c in range(len(hdr)):
        if hdr[c] != "Excitation Energy (eV):":
            continue
        e = float(hdr[c + 1])
        x = [r[c] for r in rows[2:] if c < len(r) and r[c] is not None]
        yv = [r[c + 1] for r in rows[2:] if c + 1 < len(r) and r[c] is not None]
        y, lo, hi = sur_grille(x, yv)
        out.append((y, dict(source=f"Southampton_{nom}_2021", materiau="MoSe2/WSe2",
                            structure=f"hétérobicouche ({nom}), encapsulée hBN", temperature_K=4,
                            excitation_nm=round(HC_EV_NM / e, 2), excitation_eV=round(e, 4),
                            cm1_min=round(lo, 1), cm1_max=round(hi, 1), traite_par_auteurs=True)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws2", required=True)
    ap.add_argument("--hs20", required=True)
    ap.add_argument("--hs57", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = lire_ws2(a.ws2) + lire_hs(a.hs20, "HS20") + lire_hs(a.hs57, "HS57")
    X = np.vstack([d[0] for d in data])
    lab = pd.DataFrame([d[1] for d in data])
    lab.insert(0, "famille", "TMD")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "spectres_reels.npz", X=X, grid=GRID)
    lab.to_csv(out / "etiquettes_reels.csv", index=False)
    print("Spectres réels :", len(lab))
    print(lab.groupby(["source", "temperature_K"]).size().to_string())
    print(lab.groupby("source")[["cm1_min", "cm1_max", "excitation_nm"]].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
