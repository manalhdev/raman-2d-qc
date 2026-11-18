# Dataset Raman v1 — PI² n°156

Premier dataset du projet *AI-Assisted Raman Analysis for 2D Electronic Device Quality Control*.
Construit le 17/11/2026 à partir des deux sources transmises par l'encadrante.

## Contenu

```
data/v1/
  spectres_3familles.csv     133 spectres (carbone, h-BN, TMD) x 3101 points  <- fichier principal
  metadata_tous.csv          1 ligne par matériau (3504) : formule, famille, groupe d'espace, ...
  modes_raman_tous.csv       les raies brutes calculées (fréquence, intensité) des 3504 matériaux
  spectres_tous.npz          les 3504 spectres (numpy), y compris la classe « autre »
  zenodo_MoS2_mesures.csv    586 pixels de MoS2 mesurés : positions E2g / A1g, Δω, contrainte, dopage
scripts/
  extract_materials_cloud.py   construit les 4 premiers fichiers
  extract_zenodo_features.py   construit zenodo_MoS2_mesures.csv
  figures_controle.py          figures de vérification
figures/
  spectres_exemples.png, repartition_familles.png
```

## Dataset v2 (équilibré, prêt pour l'apprentissage)

Voir `data/v2/README_v2.md` : 3750 spectres simulés « comme mesurés », 1250 par famille, séparés en train / test.

## Dataset v3 (simulé + 521 spectres mesurés)

Voir `data/v3/README_v3.md` : le v2 plus 521 vrais spectres de TMD (Université de Southampton, CC BY 4.0), dans un seul fichier `spectres_v3.npz`, avec un jeu `test_reel` pour tester les modèles sur des mesures.

## Reproduire

```bash
pip install numpy pandas spglib matplotlib scikit-learn scipy openpyxl
python scripts/extract_materials_cloud.py --zip data/sources/materials_cloud_2024.148_data.zip    --out data/v1/
python scripts/extract_zenodo_features.py --zip data/sources/zenodo_17311728_Dataset.zip --out data/v1/
python scripts/figures_controle.py        --dataset data/v1/ --out figures/
python scripts/augmenter_dataset.py       --dataset data/v1/ --out data/v2/ --seed 0
python scripts/verifier_v2.py             --dataset data/v2/ --out figures/
python scripts/extract_soton.py --ws2 data/sources/southampton_D0371_WS2_RRS_Spectra.xlsx --hs20 data/sources/southampton_D1747_HS20_Data.xlsx --hs57 data/sources/southampton_D1747_HS57_Data.xlsx --out data/v3/
python scripts/verifier_v3.py             --v2 data/v2/ --v3 data/v3/ --out figures/
python scripts/assembler_v3.py            --v2 data/v2/ --v3 data/v3/
```
`data.zip` : https://archive.materialscloud.org/record/2024.148 — `Dataset.zip` : https://zenodo.org/records/17311728

## Comment les spectres sont construits

1. **Source** : Materials Cloud Archive 2024.148 (Li et al., 2024), spectres Raman *calculés* (DFPT, code FHI-aims)
   de 3504 matériaux 2D. Chaque `raman.dat` est une **liste de raies** (fréquence en cm⁻¹, intensité en Å⁴/amu),
   pas une courbe.
2. **Famille** (règle sur la composition chimique) :
   - `carbone` : uniquement du C ;
   - `h-BN` : B et N en proportion 1:1 ;
   - `TMD` : un métal de transition M + chalcogène(s) X (S, Se, Te) en proportion MX₂ (Janus MXY inclus) ;
   - `autre` : tout le reste.
3. **Courbe** : chaque raie devient une lorentzienne de largeur à mi-hauteur **8 cm⁻¹** (paramètre `--fwhm`,
   hypothèse de notre part), puis somme des raies.
4. **Grille commune** : 100 à 3200 cm⁻¹, pas de 1 cm⁻¹ → 3101 points. Colonnes `100`, `101`, … `3200`.
5. **Normalisation** : chaque spectre est divisé par son maximum (pic le plus intense = 1).

## Ce qu'il faut savoir avant de s'en servir

- **Classes très déséquilibrées** : 127 TMD, 3 carbone, 3 h-BN. Impossible d'entraîner un classifieur
  sérieux sur 3 exemples par classe : il faudra de l'augmentation de données ou d'autres sources.
- **Polymorphes** : une même formule peut apparaître plusieurs fois (ex. MoS₂ : id 2192 en R3m, id 3103 en P2₁/m).
  Les 3 « carbone » ne sont pas tous du graphène : seul l'id 577 (P6/mmm, 2 atomes) l'est (543 et 1849 sont
  d'autres structures de carbone 2D). De même, seul l'id 1 (P-6m2) est du h-BN hexagonal ; 83 (P2₁/c) et
  3278 (P4/nmm) sont d'autres structures BN. À trier avec l'encadrante.
- **Écart calcul / expérience** : les fréquences calculées sont décalées par rapport aux mesures.
  Graphène : G calculé à **1443 cm⁻¹** contre ≈1580 mesuré (−9 %). h-BN : 1356 contre ≈1366.
  MoS₂ (id 2192) : attention, c'est du MoS₂ **massif** (polytype 3R, maille sans vide), pas une monocouche ;
  modes à ≈361-366 et 396 cm⁻¹ contre ≈382 et ≈408 mesurés sur du massif (Lee et al., 2010).
  La base ne contient aucune monocouche de MoS₂ 2H.
  Un modèle entraîné sur ces spectres et testé sur de vrais spectres sera pénalisé : c'est un argument fort
  pour les tests de décalage (étape 5) et pour un recalage.
- **Pas de bandes D et 2D** pour le graphène : ce sont des processus du second ordre, absents de ces calculs.
  Ce dataset ne sert donc pas pour l'indicateur de qualité (étape 6).
- **Spectres vides** : 5 matériaux n'ont aucun mode actif ; 3 dossiers n'ont pas de `raman.dat`
  (colonne `raman_dat_present`). Ils sont exclus de `spectres_3familles.csv`.
- **Zenodo** : les spectres bruts (`Raw Maps/*.l6m`) sont au format binaire propriétaire LabSpec 6 (Horiba)
  et ne sont pas lisibles sans ce logiciel. On n'a extrait que les **grandeurs déjà ajustées** par les auteurs.
  Médianes : monocouche E2g 384,5 / A1g 403,8 / Δω 19,4 cm⁻¹ ; bicouche 383,5 / 405,4 / 22,0 cm⁻¹,
  cohérent avec Lee et al. (2010).

## Licences et citations

- Li, G. et al., *High-throughput computation of ab initio Raman spectra for two-dimensional materials*,
  Materials Cloud Archive 2024.148 (2024), DOI 10.24435/materialscloud:bq-vp — CC BY 4.0.
- Kurtash, *Twisted MoS₂ Bilayers as Functional Elements in Memtransistors… Dataset*, Zenodo (2025),
  DOI 10.5281/zenodo.17311728 — CC BY 4.0.
