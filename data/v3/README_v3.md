# Dataset v3 — simulé (v2) + 521 spectres Raman MESURÉS

Construit le 18/11/2026. Ajoute au dataset v2 les premiers **vrais spectres** du projet, pour tester les modèles
sur des mesures réelles.

## Fichiers

| Fichier | Contenu |
|---|---|
| `spectres_v3.npz` | `X` : 4271 spectres x 3101 points (100 → 3200 cm⁻¹, pas de 1) ; `grid`. **NaN = pas de mesure à ce point** |
| `etiquettes_v3.csv` | 1 ligne par spectre, même ordre que `X` (voir colonnes ci-dessous) |
| `spectres_reels.npz`, `etiquettes_reels.csv` | Les 521 spectres mesurés seuls, avec toutes leurs métadonnées |
| `zenodo_MoS2_mesures.csv` | 586 points de mesure de MoS₂ (positions des pics, contrainte, dopage), repris du v1 |
| `plis_materiaux.csv`, `parametres_simulation.json` | Repris du v2 |
| `test_modele_reference_sur_reel.csv` | Prédictions du modèle de référence sur chaque spectre réel (voir plus bas) |

Colonnes de `etiquettes_v3.csv` : `type` (simule / reel), `source`, `split` (cv / test_reel), `groupe` (le matériau), `fold` (pli 0-2 ; -1 pour le réel), `famille`,
`materiau`, `id_materials_cloud`, `structure`, `temperature_K`, `excitation_nm`, `cm1_min`, `cm1_max`
(plage réellement mesurée), puis les paramètres de simulation pour les spectres simulés.

```python
import sys; sys.path.insert(0, "scripts")
from pretraitement import charger
from sklearn.model_selection import PredefinedSplit
X, y, groupes, plis, lab, grille = charger("data/v3", jeu="cv")          # 3750 simulés, plis groupés
Xr, yr, *_ = charger("data/v3", jeu="test_reel")                         # 521 mesurés
cv = PredefinedSplit(plis)    # validation GROUPÉE par matériau, jamais aléatoire
```
`charger` applique le prétraitement recommandé : normalisation sans le pic du silicium, fenêtres 100-1800 et
2500-2850 cm⁻¹ (2052 points), NaN mis à 0 après normalisation.

## Contenu

|  | carbone | h-BN | TMD | Total |
|---|---|---|---|---|
| cv (simulé, 3 plis groupés par matériau) | 1250 (3 matériaux) | 1250 (3) | 1250 (127) | 3750 |
| **test_reel (mesuré)** | 0 | 0 | **521** | 521 |

⚠️ Avec seulement 3 matériaux de carbone et 3 de h-BN, ces deux familles ne se généralisent pas à un matériau
nouveau : 3 % et 2 % en validation groupée (voir `README_v2.md`, correction du 18/11/2026).

## Les spectres mesurés ajoutés (Université de Southampton, licence CC BY 4.0)

| Source | Matériau | Spectres | Conditions |
|---|---|---|---|
| McDonnell et al. (2018), DOI 10.5258/SOTON/D0371 | WS₂ monocouche | 289 | 9 températures de 4 à 295 K, laser 573 à 657 nm, plage mesurée commençant entre 7 et 164 cm⁻¹ et finissant entre 608 et 858 cm⁻¹ |
| Viner, McDonnell, Smith (2021), DOI 10.5258/SOTON/D1747 — HS20 | Hétérobicouche MoSe₂/WSe₂ encapsulée hBN | 102 | 4 K, laser 549 à 780 nm, de 70-96 à 630-650 cm⁻¹ |
| idem — HS57 | Hétérobicouche MoSe₂/WSe₂ encapsulée hBN | 130 | 4 K, laser 548 à 780 nm, de 0-60 à 588-1486 cm⁻¹ |

À savoir sur ces spectres :
- Ce sont des mesures de **Raman de résonance** (laser accordé sur les excitons du matériau), souvent à **très basse
  température**. Ils diffèrent d'une mesure de routine à 532 nm et à température ambiante : certains pics y sont
  renforcés, et le WS₂ montre un **fond très fort** (photoluminescence).
- Les spectres WS₂ sont **bruts** (seule correction : efficacité du spectromètre). Les spectres HS20 / HS57 ont été
  **traités par les auteurs** (fond soustrait, intensité calibrée).
- Les intensités sont dans les unités de chaque source : il faut normaliser avant de comparer.
- Le pic du silicium (520 cm⁻¹) est visible dans beaucoup de spectres.
- La plage mesurée est courte (en général jusqu'à 600-860 cm⁻¹) : les points hors plage sont NaN.

## Premier test sur le réel (modèle de référence, pas un résultat du projet)

Random Forest entraîné sur tout le simulé, restreint à 100-620 cm⁻¹ (la plage commune avec les mesures),
puis appliqué aux 521 spectres réels. Bonne réponse = « TMD ».

| Spectres réels | Sans prétraitement | Avec prétraitement minimal* |
|---|---|---|
| WS₂ (bruts) | **11 %** | **65 %** |
| MoSe₂/WSe₂ HS20 (traités) | 82 % | 81 % |
| MoSe₂/WSe₂ HS57 (traités) | 82 % | 85 % |

\* ligne de base estimée par un minimum glissant lissé, soustraite, puis normalisation. Simple démonstration.

Ce qu'il faut en retenir :
1. **Un modèle entraîné sur du simulé échoue sur de vrais spectres bruts** (11 % sur WS₂) : le fond de
   photoluminescence le trompe.
2. **Un prétraitement même simple change tout** (11 % → 65 %) : l'étape 1 du sujet est décisive.
3. Sur 100-620 cm⁻¹, carbone et h-BN n'ont aucun pic propre (G à 1580, h-BN à 1367) : ces chiffres mesurent
   surtout « TMD ou pas TMD ». La normalisation sans silicium ne change pas ces scores.

## Limites qui restent

- Spectres réels : **uniquement des TMD**. Toujours aucun vrai spectre de graphène ni de h-BN.
- Pas de MoS₂ monocouche en spectres complets (seulement les positions de pics Zenodo).
- Pas de graphène avec défauts pour l'indicateur de qualité (étape 6).
- Monocouches MoSe₂ et WSe₂ disponibles mais pas encore intégrées : fichiers `MoSe2_Data.xlsx` et `WSe2_Data.xlsx`
  sur https://eprints.soton.ac.uk/449632.

## Reproduire

```bash
python scripts/extract_soton.py --ws2 data/sources/southampton_D0371_WS2_RRS_Spectra.xlsx --hs20 data/sources/southampton_D1747_HS20_Data.xlsx --hs57 data/sources/southampton_D1747_HS57_Data.xlsx --out data/v3/
python scripts/verifier_v3.py --v2 data/v2/ --v3 data/v3/ --out figures/
python scripts/assembler_v3.py --v2 data/v2/ --v3 data/v3/
```

## Citations

- McDonnell, L. et al. (2018). *Data for Probing Excitons, Trions and Dark Excitons in Monolayer WS₂ using Resonance
  Raman Spectroscopy*. University of Southampton. doi:10.5258/SOTON/D0371. Article : Nano Letters, doi:10.1021/acs.nanolett.7b05184.
- Viner, J., McDonnell, L., Smith, D. (2021). *Data to Support Paper on Excited Rydberg States in TMD Heterostructures*.
  University of Southampton. doi:10.5258/SOTON/D1747.
