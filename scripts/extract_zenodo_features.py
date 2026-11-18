"""
Extraction des grandeurs Raman mesurées sur MoS2 du jeu Zenodo 17311728 (Kurtash, 2025).

Source : https://zenodo.org/records/17311728 (DOI 10.5281/zenodo.17311728), licence CC BY 4.0

Attention : ce jeu ne contient PAS de spectres lisibles directement.
 - "Raw Maps/*.l6m" : cartographies brutes au format binaire propriétaire LabSpec 6 (Horiba),
   illisibles sans le logiciel LabSpec.
 - "Processed BR-spectroscopy maps/" : cartes de grandeurs déjà ajustées, pixel par pixel :
   position E2g, position A1g, écart Δω, contrainte, densité de porteurs, séparées en
   monocouche (1L) et bicouche (2L).

Ce script rassemble ces cartes en un seul tableau : une ligne par pixel mesuré.
Usage :
    python extract_zenodo_features.py --zip Dataset.zip --out dataset/
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

GRANDEURS = {
    "E_position": "E2g_cm1",
    "A_position": "A1g_cm1",
    "delta_w": "delta_w_cm1",
    "Strain": "contrainte_pct",
    "Carrier concentration": "densite_porteurs_cm2",
}
PATTERN = re.compile(r"^(E_position|A_position|delta_w|Strain|Carrier concentration)_(p\d+)_(\d+)V_(1L|2L)\.csv$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    tables = {}
    for name in zf.namelist():
        m = PATTERN.match(name.split("/")[-1])
        if not m or "Processed" not in name:
            continue
        grandeur, device, vgs, couche = m.groups()
        df = pd.read_csv(io.BytesIO(zf.read(name)), index_col=0)
        long = df.stack().reset_index()
        long.columns = ["y_um", "x_um", GRANDEURS[grandeur]]
        long["x_um"] = long["x_um"].astype(float)
        key = (device, int(vgs), couche)
        tables.setdefault(key, []).append(long.set_index(["y_um", "x_um"]))

    rows = []
    for (device, vgs, couche), parts in sorted(tables.items()):
        t = pd.concat(parts, axis=1).reset_index()
        t.insert(0, "couche", couche)
        t.insert(0, "Vgs_V", vgs)
        t.insert(0, "dispositif", device)
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["E2g_cm1", "A1g_cm1"], how="all")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.out) / "zenodo_MoS2_mesures.csv", index=False)
    print("Pixels :", len(out))
    print(out.groupby(["dispositif", "Vgs_V", "couche"]).size().to_string())
    print(out.groupby("couche")[["E2g_cm1", "A1g_cm1", "delta_w_cm1"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
