# Les 3 versions du dataset et les résultats des entraînements

Mis à jour le 23/11/2026 (colonne `type_donnee` ajoutée le soir même). Tous les chiffres viennent de fichiers du dépôt :
`resultats/comparaison_modeles.csv` (6 modèles de l'équipe, `scripts/comparer_modeles.py`) et
`resultats/shortlist_v3/` (modèles de la shortlist, `notebooks/Modeles_Raman_v3.ipynb`).

## 1. Les 3 versions en bref

| | v1 | v2 | v3 |
|---|---|---|---|
| Date | novembre 2026 (1er commit) | 18/11 (plis groupés ajoutés le 18/11) | 18/11, complété le 23/11 |
| Contenu | 133 spectres **calculés** (1 par matériau) + métadonnées de 3504 matériaux + 586 mesures MoS₂ (positions de pics) | 3750 spectres **simulés « comme mesurés »** à partir du v1 (1250 par famille) | v2 + **528 spectres mesurés** |
| Type | théorique (DFT) | théorique (DFT + simulation de mesure) | théorique + **expérimental** |
| Familles | carbone 3, h-BN 3, TMD 127 matériaux | idem (mêmes matériaux), classes équilibrées | idem + réel : 524 TMD, 4 graphite, 0 h-BN |
| Bruit, fond, Si | non | oui : largeur 4-20 cm⁻¹, décalage ±10 cm⁻¹, bruit, fond de fluorescence, pic du Si | idem pour le simulé ; vrais défauts pour le réel |
| Validation | 1 spectre par matériau : toute validation croisée est groupée | colonne `fold` : 3 plis groupés par matériau | `split = cv` (simulé, plis) / `test_reel` (mesuré) |
| Fichiers | `data/v1/` | `data/v2/` | `data/v3/` |

### Ce qui change d'une version à l'autre

- **v1 → v2** : on passe de raies calculées « parfaites » à des spectres qui ressemblent à une mesure. Les
  positions de quelques pics sont **recalées** sur des valeurs mesurées publiées (graphène G 1443 → 1580 cm⁻¹,
  h-BN 1356 → 1367, WS₂ 340/403 → 356/418). Les autres TMD ne sont pas recalés.
- **v2 → v3** : ajout des premiers spectres mesurés, pour tester les modèles sur du vrai.
  - 18/11 : 521 spectres Southampton (WS₂ monocouche brut ; hétérobicouches MoSe₂/WSe₂ traitées par les auteurs).
    Raman de résonance, 4 à 295 K, lasers 548 à 780 nm, plage courte (100 à 600-860 cm⁻¹).
  - 23/11 : 7 spectres à 532 nm et température ambiante, plage large : 6 RRUFF (3 graphite, 2 MoS₂, 1 WS₂,
    trouvés par Meriam) et 1 graphite Mendeley (trouvé par Pierre-Louis). Premiers spectres réels de carbone.
    Script : `scripts/ajouter_reels_externes.py`. Les 4271 premières lignes du v3 sont inchangées.

### Théorique ou expérimental : classement de toutes les sources

| Catégorie | Source | N | Familles | Usage |
|---|---|---|---|---|
| Théorique, calculé (DFT) | Materials Cloud 2024.148 | 133 | carbone, h-BN, TMD | base des simulations |
| Théorique, simulé | v2 (augmentation du v1) | 3750 | carbone, h-BN, TMD | entraînement et validation groupée |
| Expérimental brut, résonance | Southampton WS₂ 2018 | 289 | TMD | test réel (le plus dur : fond de photoluminescence) |
| Expérimental traité par les auteurs | Southampton HS20 + HS57 2021 | 232 | TMD | test réel |
| Expérimental brut, 532 nm | RRUFF (6) + Mendeley (1) | 7 | carbone (4), TMD (3) | test réel plage complète |
| Expérimental, positions de pics seules | Zenodo MoS₂ | 586 | TMD | écart théorie / mesure |

Tableau généré : `resultats/shortlist_v3/catalogue_donnees.csv`.

### Colonne `type_donnee` dans le v3 (demande du tuteur)

`data/v3/etiquettes_v3.csv` a deux colonnes en plus, juste après `source` (ajoutées par `scripts/assembler_v3.py`) :

- `type_donnee` : `theorique` (calcul DFT + simulation de mesure) ou `mesure` (spectre expérimental) ;
- `detail_donnee` : `calcul DFT + simulation de mesure`, `mesure brute` ou `mesure traitee par les auteurs`.

| type_donnee | detail_donnee | carbone | h-BN | TMD | Total |
|---|---|---|---|---|---|
| theorique | calcul DFT + simulation de mesure | 1250 (3 matériaux) | 1250 (3) | 1250 (127) | 3750 |
| mesure | mesure brute | 4 (graphite) | 0 | 292 (WS₂ 290, MoS₂ 2) | 296 |
| mesure | mesure traitee par les auteurs | 0 | 0 | 232 (MoSe₂/WSe₂) | 232 |

Par matériau (`resultats/shortlist_v3/comptage_par_materiau_et_type.csv`, 75 formules) : seuls **WS₂ et MoS₂**
ont à la fois du théorique (20 spectres simulés chacun, 2 structures) et du mesuré (290 et 2 spectres). Le graphite
mesuré n'a pas d'équivalent exact dans le théorique (les 3 carbones calculés sont le graphène et 2 polymorphes), et
l'hétérobicouche MoSe₂/WSe₂ mesurée n'existe qu'en monocouches séparées dans le théorique. Aucun h-BN mesuré.

### Écart entre théorie et mesure

- **Positions de pics** : la DFT donne des fréquences trop basses de 2 à 6 %. Par exemple, MoS₂ E₂g calculé à
  363,5 cm⁻¹ contre 384,6 mesuré (Zenodo), et WS₂ 339,8 / 402,7 contre 354 / 417 mesurés (Southampton).
  Voir `resultats/shortlist_v3/ecart_theorie_mesure.csv` et `resultats/decalages_pics.csv`.
- **Forme des spectres** : un classifieur sépare simulé et mesuré avec une AUC de 0,99, même après ligne de base
  (`ecart_domaine_auc.csv`). Les mesures Southampton sont très loin du simulé (résonance, basse température,
  bords de filtre du spectromètre, rayons cosmiques).
- **Massif ou monocouche** : tous les spectres RRUFF / Mendeley sont des matériaux massifs. Les pics sont aux
  mêmes endroits que pour les monocouches, mais les rapports d'intensité et les largeurs changent.

## 2. Comment on évalue

| Protocole | Ce qu'il mesure | À retenir ? |
|---|---|---|
| Validation **aléatoire** | mêmes matériaux au train et au test : surtout de la mémorisation | non, seulement pour comparer |
| Validation **groupée** (`fold`) | matériau jamais vu à l'entraînement | **oui** |
| **Réel** | entraîné sur tout le simulé, testé sur les mesures | **oui** |

Métriques utilisées en plus de l'accuracy :

- **F1 macro** : moyenne du F1 des 3 familles, chaque famille compte pareil (l'accuracy cache une famille ratée).
- **Précision / rappel par famille** : rappel = part des vrais carbone retrouvés ; précision = part des
  « carbone » prédits qui en sont vraiment.
- **Matrices de confusion** : quelle famille est prise pour quelle autre.
- **Calibration** : ECE (écart entre confiance annoncée et taux de bonnes réponses) et score de Brier.
- **Prédiction conforme** : couverture (la vraie famille est dans l'ensemble proposé) et taille des ensembles.
- **Détection hors domaine** : AUC (1 = parfait, 0,5 = hasard).

## 3. Résultats par version (même modèle : XGBoost sur les pics, ligne de base ALS)

Fichier : `resultats/shortlist_v3/resultats_par_version.csv`.

| Entraîné sur | Testé sur | Accuracy | F1 macro | Rappel carbone | Rappel h-BN | Rappel TMD |
|---|---|---|---|---|---|---|
| v1 (133 calculés) | matériau laissé de côté (133) | 95,5 % | **0,33** | 0 % | 0 % | 100 % |
| v1 | réel 532 nm (7) | 57 % | 0,53 | 25 % | - | 100 % |
| v1 | réel Southampton (521, 100-620 cm⁻¹) | 36 % | - | - | - | 36 % |
| v2 / v3 cv (3750 simulés) | validation aléatoire | 98,9 % | 0,99 | 99,5 % | 98,4 % | 98,8 % |
| v2 / v3 cv | **validation groupée** | **73,1 %** | **0,72** | 67 % | 53 % | 99 % |
| v3 | **réel 532 nm (7)** | **7/7** | 1,00 | 4/4 | - | 3/3 |
| v3 | **réel Southampton (521)** | **86 %** | - | - | - | 86 % |

Lecture :
- Sur le v1 seul, les **95,5 % ne veulent rien dire** : le modèle répond toujours TMD (127 matériaux sur 133).
  Le F1 macro (0,33) et le rappel carbone / h-BN (0 %) le montrent tout de suite.
- L'augmentation du v2 est ce qui rend le modèle utilisable sur des mesures (36 % → 86 % sur Southampton).
- Le v3 permet enfin de mesurer l'écart simulé / réel, et depuis le 23/11 sur du carbone.

## 3.b Entraîné sur le théorique, testé sur le mesuré : métriques par type

Les modèles sont entraînés **uniquement sur `type_donnee = theorique`**. Côté théorique, le score est celui de la
validation groupée (matériaux inconnus). Côté mesuré, chaque spectre passe par le modèle adapté à sa plage, comme
dans l'outil final : XGBoost pics si le spectre va au-delà de 1700 cm⁻¹ (7 spectres), XGBoost spectre sur
100-620 cm⁻¹ sinon (521 Southampton). Fichier : `resultats/shortlist_v3/metriques_theorique_vs_mesure.csv`,
figure `08c_theorique_vs_mesure_confusion.png`.

| Type | N | Accuracy | F1 macro | Accuracy équilibrée | Rappel carbone | Rappel h-BN | Rappel TMD | Précision carbone | Précision TMD |
|---|---|---|---|---|---|---|---|---|---|
| théorique (groupé) | 3750 | 73,1 % | 0,72 | 73,1 % | 67 % | 53 % | 99 % | 0,74 | 0,80 |
| mesuré, tout | 528 | 86,4 % | 0,60 | 93,1 % | 100 % (4/4) | - | 86 % | 0,16 | 1,00 |
| mesuré brut | 296 | 82,4 % | 0,70 | 91,1 % | 100 % (4/4) | - | 82 % | 0,33 | 1,00 |
| mesuré traité par les auteurs | 232 | 91,4 % | - | 91,4 % | - | - | 91 % | - | 1,00 |

Matrices de confusion (nombre de spectres) :

| Théorique, groupé | prédit carbone | prédit h-BN | prédit TMD |
|---|---|---|---|
| vrai carbone | 841 | 400 | 9 |
| vrai h-BN | 289 | 662 | 299 |
| vrai TMD | 1 | 11 | 1238 |

| Mesuré | prédit carbone | prédit h-BN | prédit TMD |
|---|---|---|---|
| vrai carbone | 4 | 0 | 0 |
| vrai TMD | 21 | 51 | 452 |

Lecture :
- Sur le théorique, l'erreur est surtout **carbone ↔ h-BN** (les matériaux légers se ressemblent) ; les TMD sont
  presque parfaits.
- Sur le mesuré, les 4 graphites sont reconnus, mais 72 TMD réels (14 %) sont pris pour du carbone ou du h-BN.
  D'où une **précision carbone de 0,16** : quand le modèle dit « carbone » sur une mesure, il a tort 21 fois sur 25.
  Le F1 macro (0,60) le montre, l'accuracy (86 %) non.
- Le mesuré brut (fond de photoluminescence, résonance) est plus dur que le mesuré traité (82 % contre 91 % sur les TMD).
- Il n'y a pas de h-BN mesuré : on ne peut pas encore dire si le modèle reconnaît un vrai h-BN.

## 4. Résultats sur v3 : tous les modèles

### 4.1 Modèles de l'équipe (`scripts/comparer_modeles.py`, spectre moyenné par 4 cm⁻¹)

Prétraitement « ligne_base » (SNIP). Réel = 528 spectres (524 TMD + 4 graphite).

| Modèle | Acc. aléatoire | Acc. groupée | F1 macro groupée | Rappel groupé carbone / h-BN / TMD | Acc. réel | Rappel réel carbone (4) |
|---|---|---|---|---|---|---|
| KNN | 99,4 % | 59,9 % | 0,58 | 18 / 64 / 98 % | 78,8 % | 4/4 |
| Régression logistique | 99,6 % | 44,2 % | 0,36 | 6 / 27 / 99 % | 84,7 % | 4/4 |
| SVM (RBF) | 99,9 % | 33,4 % | 0,18 | 0 / 0 / 100 % | **99,4 %** | **1/4** |
| Random Forest | 99,7 % | 33,7 % | 0,20 | 1 / 0 / 100 % | 80,9 % | 4/4 |
| XGBoost | 99,7 % | 33,7 % | 0,19 | 1 / 0 / 100 % | 79,9 % | 2/4 |
| Réseau de neurones (MLP) | 99,6 % | 52,6 % | 0,47 | 39 / 19 / 100 % | 89,4 % | 4/4 |

Le prétraitement « recommande » donne des chiffres proches (voir le CSV).

**Point important** : le SVM avait 99,6 % sur le réel, mais il répond presque toujours « TMD ». Son F1 macro
groupé est de 0,18, et sur les 4 graphites réels il n'en reconnaît qu'un. Les 99,6 % venaient du fait que tout le
réel était du TMD. Avec seulement l'accuracy, on l'aurait choisi à tort.

### 4.2 Modèles de la shortlist (`notebooks/Modeles_Raman_v3.ipynb`, ligne de base ALS)

| Modèle | Acc. aléatoire | Acc. groupée | F1 macro groupée | ECE groupée | Réel Southampton (521) | Réel 532 nm (7) |
|---|---|---|---|---|---|---|
| XGBoost sur 8 pics (position, hauteur, largeur) | 98,9 % | **73,1 %** | **0,72** | 0,18 | 79 % | **7/7** |
| XGBoost sur le spectre | 99,6 % | 33,8 % | 0,20 | 0,65 | 86 % | 6/7 |
| CNN 1D (ensemble de 3) | 94,9 % | 51,0 % | 0,46 | 0,26 | **87 %** | **7/7** |
| TabPFN | non lancé : licence Hugging Face à accepter, cellule prête pour Colab | | | | | |

Précision / rappel / F1 en validation groupée (`precision_rappel_f1.csv`) :

| Modèle | Famille | Précision | Rappel | F1 |
|---|---|---|---|---|
| XGBoost pics | carbone | 0,74 | 0,67 | 0,71 |
| | h-BN | 0,62 | 0,53 | 0,57 |
| | TMD | 0,80 | 0,99 | 0,89 |
| CNN 1D | carbone | 0,59 | 0,16 | 0,25 |
| | h-BN | 0,43 | 0,48 | 0,45 |
| | TMD | 0,55 | 0,89 | 0,68 |
| XGBoost spectre | carbone | 0,08 | 0,02 | 0,03 |
| | h-BN | 0,00 | 0,00 | 0,00 |
| | TMD | 0,39 | 1,00 | 0,56 |

Matrices de confusion : `resultats/shortlist_v3/figures/04_confusion_groupe.png` (et `figures/fig06` pour les
modèles de l'équipe).

### 4.3 Robustesse sur les spectres réels

- **Ligne de base** : sans correction, le WS₂ brut est reconnu à 19 % (XGBoost spectre) ; avec ALS, 82 %.
  Figure `07_reel_modele_pretraitement.png`.
- **Température** (WS₂ brut) : 93 % entre 4 et 75 K, 67 % à 295 K (figure `08_robustesse_reel.png`).
- **Laser** : quelques longueurs d'onde ratées (HS57 vers 600 nm, HS20 à 780 nm) : en résonance, les rapports
  d'intensité des pics changent.
- **Réel 532 nm, spectre par spectre** (`reel_532nm_par_spectre.csv`, figure `08b_reel_532nm_plage_complete.png`) :
  XGBoost pics et CNN reconnaissent les 4 graphites et les 3 TMD. XGBoost spectre prend le graphite avec défauts
  (R120025) pour du h-BN.

### 4.4 Calibration et prédiction conforme

- En validation groupée, les modèles sont **trop sûrs d'eux** : XGBoost pics annonce 91 % de confiance en
  moyenne pour 73 % de bonnes réponses ; XGBoost spectre annonce 98 % pour 34 % (`f1_calibration.csv`, figure
  `04b_f1_calibration.png`).
- Prédiction conforme à 95 % (`resultats_conforme.csv`) :

| Calibration | Couverture | Taille moyenne de l'ensemble | Ensembles vides |
|---|---|---|---|
| aléatoire (méthode habituelle), test sur matériaux inconnus | 59 % | 0,67 | 33 % |
| **groupée** (un pli de matériaux non vus), test sur matériaux inconnus | **91 %** | 1,88 | 0 % |
| aléatoire, test réel Southampton | 87 % | 1,11 | 0 % |
| groupée, test réel Southampton | 88 % | 1,14 | 0 % |

Calibrer sur des **matériaux** non vus remet la garantie presque à 95 %. Le prix : des ensembles plus grands
(le modèle dit « carbone ou h-BN » quand il ne sait pas).

### 4.5 Détection hors domaine (`resultats_anomalies.csv`)

| Question | Isolation Forest | Plus proche voisin (cosinus) |
|---|---|---|
| Famille jamais vue (entraîné sans TMD) | AUC 0,63 | **AUC 1,00** |
| Spectre mesuré vs simulé | AUC 0,83 | **AUC 0,96** |

## 5. Pourquoi de tels écarts ?

1. **Trop peu de matériaux de carbone et de h-BN.** Chaque pli n'en a que 2 à l'entraînement. En faisant varier
   le nombre de matériaux TMD vus, le rappel TMD groupé passe de 44 % (2 matériaux) à 93 % (20) et 99 % (80)
   (`courbe_nb_materiaux.csv`, figure `06_courbe_materiaux.png`). Il faudrait ~20 matériaux par famille.
2. **Les matériaux carbone / h-BN du jeu sont très différents entre eux** (polymorphes calculés). Avec XGBoost
   pics, le graphène (id 577) est reconnu à 97 % quand il est mis de côté, mais la monocouche de h-BN (id 1) à
   seulement 2 % : son pic unique vers 1367 cm⁻¹ ressemble à un pic de carbone, et les deux autres h-BN du jeu
   n'ont pas de pic à cet endroit. Or c'est justement le h-BN qui compte pour le projet.
3. **Spectre brut vs pics** : sur le spectre complet, les modèles apprennent par cœur la forme exacte des 2
   matériaux vus (score groupé ≈ 33 %, c'est-à-dire « tout est TMD »). Les positions et largeurs de pics sont
   des grandeurs physiques qui se transfèrent mieux (73 %).
4. **Simulé vs réel** : les mesures Southampton sont très particulières (résonance, basse température). Sur des
   mesures à 532 nm et à température ambiante, proches d'un contrôle de routine, les modèles plage complète
   s'en sortent bien (7/7), mais sur seulement 7 spectres.

## 6. Prochaines étapes

1. Données : plus de matériaux carbone et h-BN, et surtout de vrais spectres de graphène et de h-BN monocouche
   à 532 nm. Recaler les TMD matériau par matériau.
2. Toujours présenter accuracy + F1 macro + rappel par famille + test réel, jamais l'accuracy seule.
3. Modèles : XGBoost pics comme référence, CNN + ALS pour le réel, prédiction conforme calibrée par matériau,
   alerte hors domaine par plus proche voisin.
