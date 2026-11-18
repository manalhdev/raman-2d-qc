# Dataset v2 — spectres simulés « comme mesurés », classes équilibrées

Construit le 17/11/2026 à partir du dataset v1 par `scripts/augmenter_dataset.py`.
Reproductible : `python scripts/augmenter_dataset.py --dataset data/v1/ --out data/v2/ --seed 0`

## Fichiers

| Fichier | Contenu |
|---|---|
| `spectres_v2.npz` | `X` : 3750 spectres x 3101 points (coups simulés, **non normalisés**) ; `grid` : 100 → 3200 cm⁻¹ |
| `etiquettes_v2.csv` | 1 ligne par spectre, même ordre que `X` : `split`, `id`, `formule`, `famille`, `recale`, et les paramètres tirés (`fwhm`, `decalage`, `silicium`, `fond`, `snr`) |
| `split_materiaux.csv` | Pour chaque matériau de base : train, test ou train+test |
| `parametres_simulation.json` | Tous les paramètres utilisés (pour la reproductibilité et le rapport) |

Chargement :
```python
import numpy as np, pandas as pd
d = np.load("data/v2/spectres_v2.npz"); X, grid = d["X"], d["grid"]
lab = pd.read_csv("data/v2/etiquettes_v2.csv")
X_train, y_train = X[lab.split == "train"], lab.famille[lab.split == "train"]
X_test,  y_test  = X[lab.split == "test"],  lab.famille[lab.split == "test"]
```

## Répartition

|  | carbone | h-BN | TMD |
|---|---|---|---|
| train | 1000 | 1000 | 1000 |
| test | 250 | 250 | 250 |

## Ce qui a été fait, dans l'ordre

1. **Séparation par formule (TMD)** : 14 formules sur 71 (≈ 20 %) sont réservées au test. Aucun spectre
   d'une formule de test n'est vu à l'entraînement : le test mesure la capacité à reconnaître un TMD *inconnu*.
2. **Carbone et h-BN : pas de séparation possible** (une seule formule chacun : C et BN). Les mêmes matériaux
   de base servent au train et au test, avec des variantes tirées indépendamment. Le score sur ces deux
   classes mesure donc la **robustesse aux conditions de mesure**, pas la généralisation à un nouveau matériau.
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
- Classifieur de référence (Random Forest, spectres simplement ramenés entre 0 et 1) : **96 % de bonnes réponses**
  sur le test. Ce n'est **pas un résultat du projet**, seulement la preuve que le dataset est apprenable.
  Le chiffre est optimiste : spectres simulés, et pour carbone et h-BN, mêmes matériaux de base au train et au test.

## Limites à connaître

- **Pas de monocouche de MoS₂ 2H** dans la base : l'id 2192 est du MoS₂ **massif** (polytype 3R, maille sans vide),
  l'id 3103 est la phase 1T′. Le TMD le plus courant manque donc.
- Dans la classe carbone, 2 matériaux de base sur 3 ne sont pas du graphène (ids 543 et 1849, autres structures de carbone).
  Dans la classe h-BN, 2 sur 3 ne sont pas du h-BN hexagonal (ids 83 et 3278). Ils représentent environ 2/3 des
  spectres de ces classes. À décider en équipe : les garder ou n'utiliser que le graphène et le h-BN.
- Les spectres restent **simulés** : il faut de vrais spectres pour le test final.
- Pas encore de graphène avec défauts (bande D) : prévu pour l'étape 6.

## Sources des valeurs de recalage

- Ferrari & Basko, *Raman spectroscopy as a versatile tool for studying the properties of graphene*, arXiv:1306.5856.
- Gorbachev et al., *Hunting for monolayer boron nitride: optical and Raman signatures*, arXiv:1008.2868 (Small, 2011).
- WS₂ : arXiv:1805.12287 (≈ 356 et 417 cm⁻¹) ; arXiv:1610.06813 (355,7 et 418,4 cm⁻¹).
