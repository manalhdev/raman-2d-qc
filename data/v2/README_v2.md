# Dataset v2 — spectres simulés « comme mesurés », classes équilibrées, plis groupés par matériau

Construit le 17/11/2026 à partir du dataset v1 par `scripts/augmenter_dataset.py`.
Reproductible : `python scripts/augmenter_dataset.py --dataset data/v1/ --out data/v2/ --seed 0`

## Fichiers

| Fichier | Contenu |
|---|---|
| `spectres_v2.npz` | `X` : 3750 spectres x 3101 points (coups simulés, **non normalisés**) ; `grid` : 100 → 3200 cm⁻¹ |
| `etiquettes_v2.csv` | 1 ligne par spectre, même ordre que `X` : `groupe` (le matériau), `fold` (0, 1 ou 2), `id`, `formule`, `famille`, `recale`, et les paramètres tirés (`fwhm`, `decalage`, `silicium`, `fond`, `snr`) |
| `plis_materiaux.csv` | Pour chaque matériau de base : son pli |
| `parametres_simulation.json` | Tous les paramètres utilisés (pour la reproductibilité et le rapport) |

Chargement recommandé (prétraitement et validation groupée) :
```python
import sys; sys.path.insert(0, "scripts")
from pretraitement import charger
from sklearn.model_selection import PredefinedSplit
X, y, groupes, plis, lab, grille = charger("data/v2")   # 3750 x 2052 points (fenêtres utiles)
cv = PredefinedSplit(plis)                                  # 3 plis : un matériau ne change jamais de pli
```

## ⚠️ Correction du 18/11/2026 : fuite par matériau

La première version séparait entraînement et test par formule. Or le carbone et le h-BN n'ont que **3 matériaux
chacun**, et une seule formule (C, BN) : les **mêmes matériaux** étaient en entraînement et en test. Un modèle
reconnaissait donc des matériaux déjà vus, pas une famille.

| Validation (Random Forest de référence) | Global | Carbone | h-BN | TMD |
|---|---|---|---|---|
| Aléatoire (fuite par matériau) | 97,5 % | 97 % | 97 % | 98 % |
| **Groupée par matériau** | **34,7 %** | **3 %** | **2 %** | **99 %** |

Le bon score ne vaut que pour les TMD (127 matériaux). Avec 3 matériaux, le carbone et le h-BN ne se généralisent
pas à un matériau jamais vu. **Il faut ajouter des matériaux de carbone et de h-BN** (décision d'équipe).

Ce qui change :
- plus de séparation fixe train / test : chaque spectre porte son **matériau** (`groupe`) et son **pli** (`fold`),
  et tous les spectres d'un matériau sont dans le même pli ; on valide en **GroupKFold par matériau** ;
- 3 plis, pour que chaque pli contienne un matériau de carbone et un de h-BN ;
- 1250 spectres par famille, répartis à parts égales entre les matériaux de la famille ;
- prétraitement recommandé (`scripts/pretraitement.py`) : normalisation **sans le pic du silicium**
  (505-535 cm⁻¹ ignorés pour le maximum) et **fenêtres 100-1800 et 2500-2850 cm⁻¹** seulement ;
- pas d'imputation ni de SMOTE : ni NaN, ni doublon, classes équilibrées ; moyenner deux spectres créerait des pics doublés.

## Ce qui a été fait, dans l'ordre

1. **Plis groupés par matériau** : dans chaque famille, les matériaux sont répartis à tour de rôle entre 3 plis.
2. **Variantes** : 1250 spectres par famille, autant par matériau.
3. **Recalage** des positions calculées sur des valeurs mesurées publiées :
   - graphène (id 577) : G 1443 → **1580** cm⁻¹ (Ferrari & Basko, 2013) ;
   - h-BN monocouche (id 1) : 1356 → **1367** cm⁻¹ (1366 en massif, décalage de +1 à +4 en monocouche : Gorbachev et al., 2011) ;
   - WS₂ monocouche (id 1348) : E′ 340 → **356**, A₁′ 403 → **418** cm⁻¹ (valeurs publiées ≈ 356 et 417-418).
   Les autres matériaux ne sont pas recalés : la tolérance vient du décalage aléatoire (point 4).
4. **Simulation de mesures réelles** (paramètres par défaut, **à valider en équipe**) :
   - largeur des pics 4 à 20 cm⁻¹, ±20 % par pic ;
   - décalage global ±10 cm⁻¹ (calibration + erreur du calcul), ±3 cm⁻¹ par pic (contrainte, dopage) ;
   - intensité relative de chaque pic multipliée par un facteur aléatoire (lognormal, σ = 0,3) ;
   - pic du silicium à 520,5 cm⁻¹ dans 50 % des spectres ;
   - fond (polynôme, fluorescence exponentielle ou bosse large) dans 80 % des spectres ;
   - bruit : rapport signal sur bruit entre 15 et 300 ;
   - graphène : ajout de la **bande 2D** (absente du calcul) vers 2670-2700 cm⁻¹, largeur 24-35 cm⁻¹,
     I(2D)/I(G) entre 1 et 4 (hypothèse : graphène monocouche de bonne qualité).

## Vérifications faites

- Pics recalés après simulation (médianes) : graphène 1578 cm⁻¹ (cible 1580), h-BN 1368 (cible 1367), WS₂ A₁′ 417,5 (cible 418).
- Validation groupée par matériau du Random Forest de référence : **34,7 %** (carbone 3 %, h-BN 2 %, TMD 99 %),
  contre 97,5 % en validation aléatoire. Voir la correction plus haut. Ce n'est pas un résultat du projet.

## Limites à connaître

- **Pas de monocouche de MoS₂ 2H** dans la base : l'id 2192 est du MoS₂ **massif** (polytype 3R, maille sans vide),
  l'id 3103 est la phase 1T′. Le TMD le plus courant manque donc.
- Carbone et h-BN : 3 matériaux chacun, donc impossible de généraliser à un matériau nouveau (voir la correction).
- Dans la classe carbone, 2 matériaux de base sur 3 ne sont pas du graphène (ids 543 et 1849, autres structures de carbone).
  Dans la classe h-BN, 2 sur 3 ne sont pas du h-BN hexagonal (ids 83 et 3278). Ils représentent environ 2/3 des
  spectres de ces classes. À décider en équipe : les garder ou n'utiliser que le graphène et le h-BN.
- Les spectres restent **simulés** : il faut de vrais spectres pour le test final.
- Pas encore de graphène avec défauts (bande D) : prévu pour l'étape 6.

## Sources des valeurs de recalage

- Ferrari & Basko, *Raman spectroscopy as a versatile tool for studying the properties of graphene*, arXiv:1306.5856.
- Gorbachev et al., *Hunting for monolayer boron nitride: optical and Raman signatures*, arXiv:1008.2868 (Small, 2011).
- WS₂ : arXiv:1805.12287 (≈ 356 et 417 cm⁻¹) ; arXiv:1610.06813 (355,7 et 418,4 cm⁻¹).
