"""raman_qc : pipeline d'analyse Raman pour le contrôle qualité de matériaux 2D (projet PI² n°156).

Modules
-------
donnees       chargement du dataset v3 et catalogue des sources (théorique / expérimental)
pretraitement ligne de base ALS, lissage, normalisation sans silicium, découpe en fenêtres
features      spectre moyenné par bandes et caractéristiques de pics
modeles       XGBoost, TabPFN, CNN 1D (ensemble), Isolation Forest, prédiction conforme
evaluation    validation aléatoire vs groupée par matériau, test sur spectres mesurés
"""
__version__ = "0.3.0"
