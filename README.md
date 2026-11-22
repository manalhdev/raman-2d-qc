# raman-2d-qc — AI-Assisted Raman Analysis for 2D Electronic Device Quality Control

Projet d'Innovation Industrielle (PI²) n°156 — ESILV, octobre 2026 à avril 2027.
Encadrante : Sabrine Ayari. Équipe de 5 étudiants.

**Objectif** : une plateforme d'IA qui prend un spectre Raman d'un matériau 2D et répond à deux questions :
**quel matériau est-ce** (graphène / carbone, h-BN, dichalcogénures de métaux de transition TMD) et
**est-il de bonne qualité**.

> État : construction des données (novembre 2026). Les modèles, l'indicateur de qualité et l'application web
> viendront ensuite.

## Démarche (6 étapes du sujet)

1. Prétraitement : ligne de base, débruitage, normalisation, grille commune
2. Détection des pics et extraction de caractéristiques physiques
3. Classification du matériau et de sa famille (carbone, h-BN, TMD)
4. Comparaison de modèles : KNN, SVM, Random Forest, réseau de neurones, CNN 1D
5. Tests de robustesse : bruit, décalages spectraux, variations d'intensité
6. Indicateur de qualité du graphène à partir des bandes D, G et 2D

## Contenu du dépôt

```
data/
  sources/   fichiers d'origine téléchargés (voir « Sources et licences »)
  v1/        133 spectres calculés (3 familles) + métadonnées des 3504 matériaux + mesures MoS2 (Zenodo)
  v2/        3750 spectres simulés « comme mesurés », équilibrés, en 3 plis groupés par matériau
  v3/        v2 + 521 spectres Raman MESURÉS (test_reel)   <- dataset à utiliser
scripts/     extraction, augmentation, assemblage, vérifications, modèles, figures (Python)
figures/     figures du rapport (fig01 à fig10) et figures de contrôle
resultats/   tableaux de résultats (CSV) et prédictions détaillées
demo/        application de démonstration (Streamlit) et spectres d'exemple
docs/        README_dataset.md : description détaillée des données
rapports/    rapports de suivi hebdomadaires (Word)
```

Détails : [`docs/README_dataset.md`](docs/README_dataset.md), [`data/v2/README_v2.md`](data/v2/README_v2.md),
[`data/v3/README_v3.md`](data/v3/README_v3.md).

## Démarrage rapide

```bash
pip install -r requirements.txt
```
```python
import numpy as np, pandas as pd
d = np.load("data/v3/spectres_v3.npz"); X, grid = d["X"], d["grid"]   # (4271, 3101), 100 -> 3200 cm-1
lab = pd.read_csv("data/v3/etiquettes_v3.csv")
```
Avec le prétraitement recommandé et les plis groupés par matériau :
```python
import sys; sys.path.insert(0, "scripts")
from pretraitement import charger
from sklearn.model_selection import PredefinedSplit
X, y, groupes, plis, lab, grille = charger("data/v3", jeu="cv")   # simulé : valider avec PredefinedSplit(plis)
Xr, yr, *_ = charger("data/v3", jeu="test_reel")                  # 521 spectres mesurés
```
`X` contient des NaN là où un spectre mesuré n'a pas de donnée.

## Reproduire les données

```bash
python scripts/extract_materials_cloud.py --zip data/sources/materials_cloud_2024.148_data.zip --out data/v1/
python scripts/extract_zenodo_features.py --zip data/sources/zenodo_17311728_Dataset.zip --out data/v1/
python scripts/augmenter_dataset.py --dataset data/v1/ --out data/v2/ --seed 0
python scripts/extract_soton.py --ws2 data/sources/southampton_D0371_WS2_RRS_Spectra.xlsx \
    --hs20 data/sources/southampton_D1747_HS20_Data.xlsx --hs57 data/sources/southampton_D1747_HS57_Data.xlsx --out data/v3/
python scripts/assembler_v3.py --v2 data/v2/ --v3 data/v3/
# vérifications et figures
python scripts/figures_controle.py --dataset data/v1/ --out figures/
python scripts/verifier_v2.py --dataset data/v2/ --out figures/
python scripts/verifier_v3.py --v2 data/v2/ --v3 data/v3/ --out figures/
```
Vérifié le 18/11/2026 : cette suite redonne à l'identique les fichiers de `data/v1`, `data/v2` et `data/v3`.

## Principaux constats à ce stade

- Base calculée très déséquilibrée : 127 TMD, 3 carbone, 3 h-BN ; aucune monocouche de MoS₂ 2H.
- **Fuite par matériau (corrigée le 18/11/2026)** : en validation aléatoire, un modèle de référence atteint 97,5 % ;
  groupé par matériau, il tombe à 34,7 % (carbone 3 %, h-BN 2 %, TMD 99 %). Le bon score ne vaut que pour les TMD :
  il faut plus de matériaux de carbone et de h-BN. Toujours valider en GroupKFold par matériau (colonne `fold`).
- Fréquences calculées décalées par rapport aux mesures (graphène : G à 1443 contre ≈1580 cm⁻¹) : recalage dans le v2.
- Un modèle entraîné sur des spectres simulés reconnaît 11 % des vrais spectres de WS₂ sans prétraitement,
  65 % avec une simple soustraction de ligne de base : **le prétraitement est décisif**.
- Prétraitement recommandé (`scripts/pretraitement.py`) : normalisation sans le pic du silicium, fenêtres
  100-1800 et 2500-2850 cm⁻¹ ; ni imputation ni SMOTE (pas de NaN dans le simulé, classes équilibrées).
- Pas encore de vrais spectres de graphène et de h-BN, ni de graphène avec défauts (étape 6).

## Résultats au 22/11/2026 (premier jet)

```bash
python scripts/analyse_decalages.py --racine .      # décalage théorique / mesuré -> resultats/decalages_pics.csv
python scripts/comparer_modeles.py --racine .       # 6 modèles x 2 prétraitements x 3 évaluations (~10 min)
python scripts/figures_donnees.py --racine .        # figures 01, 03, 09, 10
python scripts/figures_resultats.py --racine .      # figures 04 à 08
```

- Validation aléatoire 92-100 %, validation groupée par matériau 33-60 % : la fuite par matériau concerne tous les modèles.
- Carbone et h-BN (3 matériaux chacun) ne sont presque jamais reconnus sur un matériau nouveau : il faut plus de matériaux.
- Ligne de base SNIP + SVM : 99,6 % sur les 521 spectres mesurés (TMD seulement), contre 89,6 % sans.
- Fréquences calculées toutes trop basses de 7 à 137 cm⁻¹ (1 à 9 %), décalage différent selon le matériau et le mode.

Figures (PNG 300 dpi + SVG, style commun `scripts/style_figures.py`) : `figures/fig01` à `fig10`.

## Démo

```bash
pip install -r requirements.txt
python scripts/entrainer_modele_demo.py --racine . --modele "SVM (RBF)" --pretraitement ligne_base
streamlit run demo/app.py
```
On dépose un spectre (CSV/TXT : nombre d'onde, intensité) ; l'appli affiche la famille prédite, la confiance et les pics détectés.
Exemples dans `demo/exemples/`.

## Sources et licences des données

| Source | Contenu | Licence |
|---|---|---|
| Li, G. et al. (2024), Materials Cloud Archive 2024.148, DOI 10.24435/materialscloud:bq-vp | Spectres Raman calculés de 3504 matériaux 2D | CC BY 4.0 |
| Kurtash (2025), Zenodo, DOI 10.5281/zenodo.17311728 | Mesures de MoS₂ (positions de pics, contrainte, dopage) | CC BY 4.0 |
| McDonnell, L. et al. (2018), University of Southampton, DOI 10.5258/SOTON/D0371 | Spectres Raman mesurés de WS₂ monocouche | CC BY 4.0 |
| Viner, J., McDonnell, L., Smith, D. (2021), University of Southampton, DOI 10.5258/SOTON/D1747 | Spectres Raman mesurés d'hétérostructures MoSe₂/WSe₂ | CC BY 4.0 |
| Lafuente, B. et al. (2015), RRUFF Project, échantillons R050503, R090047, R120025 (graphite), R050209, R060124 (MoS₂), R070616 (WS₂) | 6 spectres Raman bruts mesurés sur minéraux massifs (532 nm) : 3 graphite, 2 MoS₂, 1 WS₂ | Citation demandée ; licence à vérifier |

Les fichiers de `data/sources/` sont redistribués tels quels sous leur licence d'origine ; les jeux dérivés
(`data/v1` à `data/v3`) restent soumis à l'attribution de ces sources.


## Licence du code

Code (`scripts/`) sous licence MIT, voir [`LICENSE`](LICENSE).
