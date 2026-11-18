"""Assemble le dataset v3 : spectres simulés du v2 + spectres mesurés (Southampton) dans un seul fichier.
Usage : python assembler_v3.py --v2 dataset_v2/ --v3 dataset_v3/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", required=True)
    a = ap.parse_args()
    v2, v3 = Path(a.v2), Path(a.v3)
    d2 = np.load(v2 / "spectres_v2.npz"); dr = np.load(v3 / "spectres_reels.npz")
    assert np.array_equal(d2["grid"], dr["grid"])
    l2 = pd.read_csv(v2 / "etiquettes_v2.csv")
    lr = pd.read_csv(v3 / "etiquettes_reels.csv")

    s = pd.DataFrame(dict(type="simule", source="Materials_Cloud_2024.148_augmente", split=l2.split,
                          famille=l2.famille, materiau=l2.formule, id_materials_cloud=l2.id,
                          structure="", temperature_K=np.nan, excitation_nm=np.nan,
                          cm1_min=100.0, cm1_max=3200.0))
    s = pd.concat([s, l2[["recale", "fwhm", "decalage", "silicium", "fond", "snr"]]], axis=1)
    r = pd.DataFrame(dict(type="reel", source=lr.source, split="test_reel", famille=lr.famille,
                          materiau=lr.materiau, id_materials_cloud=np.nan, structure=lr.structure,
                          temperature_K=lr.temperature_K, excitation_nm=lr.excitation_nm,
                          cm1_min=lr.cm1_min.clip(lower=100), cm1_max=lr.cm1_max.clip(upper=3200)))
    lab = pd.concat([s, r], ignore_index=True)
    X = np.vstack([d2["X"].astype(np.float32), dr["X"].astype(np.float32)])
    np.savez_compressed(v3 / "spectres_v3.npz", X=X, grid=d2["grid"])
    lab.to_csv(v3 / "etiquettes_v3.csv", index=False)
    print(X.shape)
    print(lab.groupby(["type", "split", "famille"]).size().to_string())


if __name__ == "__main__":
    main()
