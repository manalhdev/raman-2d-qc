"""Modèles de la shortlist (décision d'équipe du 18/11/2026).

1. Pics + XGBoost       baseline interprétable
2. TabPFN               modèle « fondation » tabulaire (poids sur Hugging Face, accès à demander)
3. CNN 1D (ensemble)    pré-entraîné sur le simulé, appliqué tel quel aux mesures
4. Isolation Forest     détection d'anomalies (spectre hors du domaine d'entraînement)
+  Prédiction conforme  ensemble de classes avec garantie de couverture
"""
import os

import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

SEED = 0


def xgboost(seed=SEED):
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                         colsample_bytree=0.8, n_jobs=-1, verbosity=0, random_state=seed)


def tabpfn(seed=SEED):
    """TabPFN v2. Lève une exception si les poids ne sont pas accessibles (compte Hugging Face requis)."""
    from tabpfn import TabPFNClassifier
    import torch
    return TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", random_state=seed)


def tabpfn_disponible():
    try:
        m = tabpfn()
        m.fit(np.random.rand(20, 3), np.arange(20) % 2)
        return True, ""
    except Exception as e:  # pas de réseau, pas de token, licence non acceptée...
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


def isolation_forest(seed=SEED):
    return IsolationForest(n_estimators=300, random_state=seed)


# ---------------------------------------------------------------- CNN 1D
class CNNEnsemble:
    """Ensemble de petits CNN 1D (moyenne des probabilités). Interface sklearn : fit / predict_proba / predict.

    Augmentations pendant l'entraînement (pour être robuste aux mesures réelles) : gain aléatoire, bruit,
    petit décalage en cm-1, et masquage du début du spectre (les mesures réelles commencent entre 100 et 165 cm-1).
    """

    def __init__(self, n_modeles=5, epoques=30, lr=3e-3, batch=64, n_classes=3, masque_debut=0, seed=SEED):
        self.n_modeles, self.epoques, self.lr, self.batch = n_modeles, epoques, lr, batch
        self.n_classes, self.masque_debut, self.seed = n_classes, masque_debut, seed

    @staticmethod
    def _reseau(n_classes):
        import torch.nn as nn

        def bloc(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))

        class CNN1D(nn.Module):
            def __init__(s):
                super().__init__()
                s.f = nn.Sequential(bloc(1, 16, 9), bloc(16, 32, 7), bloc(32, 64, 5), bloc(64, 64, 3),
                                    nn.AdaptiveAvgPool1d(1), nn.Flatten())
                s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n_classes))

            def forward(s, x):
                return s.h(s.f(x.unsqueeze(1)))
        return CNN1D()

    def _augmenter(self, xb, torch):
        n = len(xb)
        xb = xb * (1 + 0.1 * torch.randn(n, 1)) + 0.01 * torch.randn_like(xb)
        dec = int(torch.randint(-3, 4, (1,)))
        xb = torch.roll(xb, dec, dims=1)
        if self.masque_debut:
            k = torch.randint(0, self.masque_debut + 1, (n,))
            pos = torch.arange(xb.shape[1]).unsqueeze(0)
            xb = torch.where(pos < k.unsqueeze(1), torch.zeros_like(xb), xb)
        return xb

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(os.cpu_count() or 4)
        Xt, yt = torch.tensor(np.asarray(X), dtype=torch.float32), torch.tensor(np.asarray(y), dtype=torch.long)
        self.modeles_, self.historique_ = [], []
        pas = (len(Xt) + self.batch - 1) // self.batch
        for s in range(self.n_modeles):
            torch.manual_seed(self.seed + s)
            m = self._reseau(self.n_classes)
            opt = torch.optim.AdamW(m.parameters(), lr=self.lr, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.epoques * pas)
            h = []
            for _ in range(self.epoques):
                m.train(); perm = torch.randperm(len(Xt)); tot = 0.0
                for i in range(0, len(Xt), self.batch):
                    idx = perm[i:i + self.batch]
                    loss = F.cross_entropy(m(self._augmenter(Xt[idx], torch)), yt[idx])
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
                    tot += loss.item() * len(idx)
                h.append(tot / len(Xt))
            self.modeles_.append(m.eval()); self.historique_.append(h)
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
            return np.mean([torch.softmax(m(Xt), 1).numpy() for m in self.modeles_], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def sauver(self, chemin):
        import torch
        torch.save({"etats": [m.state_dict() for m in self.modeles_], "n_classes": self.n_classes}, chemin)


# ---------------------------------------------------------------- Prédiction conforme
class Conforme:
    """Split conformal (score LAC = 1 - proba de la vraie classe), niveau 1 - alpha.

    La garantie « la vraie classe est dans l'ensemble avec probabilité >= 1 - alpha » ne vaut que si les
    spectres testés ressemblent aux spectres de calibration (échangeabilité).
    """

    def __init__(self, modele, alpha=0.05):
        self.modele, self.alpha = modele, alpha

    def calibrer(self, Xc, yc):
        sc = 1 - self.modele.predict_proba(Xc)[np.arange(len(yc)), yc]
        n = len(sc)
        self.q_ = np.quantile(sc, min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n), method="higher")
        return self

    def ensembles(self, X):
        return (1 - self.modele.predict_proba(X)) <= self.q_

    @staticmethod
    def resume(S, y):
        taille = S.sum(1)
        return dict(couverture=float(S[np.arange(len(y)), y].mean()), taille_moyenne=float(taille.mean()),
                    singletons=float((taille == 1).mean()), vides=float((taille == 0).mean()))
