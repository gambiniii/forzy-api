"""
Os três detectores de novidade, todos treinados APENAS em dado saudável.

  Mahalanobis robusta (MinCovDet)  — baseline estatístico
  Isolation Forest (300 árvores)   — ML clássico
  Autoencoder 16→8→4→8→16 (torch)  — Deep Learning, é o escolhido

Semente 42 em três lugares (sklearn random_state, np.random.default_rng,
torch.manual_seed) para reprodutibilidade bit a bit na mesma máquina.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
class MahalanobisRobusta:
    """Distância ao centro da nuvem saudável considerando as correlações.

    A matriz de covariância é DEFICIENTE DE POSTO: dentro de um regime, features
    como v_rms e v_roll_mean são colineares. É preciso somar uma crista
    proporcional ao traço antes de inverter, e usar pinv em vez de inv.
    """

    def __init__(self, support_fraction: float = 0.85, crista: float = 1e-6):
        self.support_fraction = support_fraction
        self.crista = crista
        self.centro_: np.ndarray | None = None
        self.prec_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MahalanobisRobusta":
        X = np.asarray(X, dtype=float)
        try:
            mcd = MinCovDet(support_fraction=self.support_fraction, random_state=SEED).fit(X)
            centro, cov = mcd.location_, mcd.covariance_
        except Exception:
            # MinCovDet falha quando o posto é muito baixo (regime PARADO, em que
            # várias features são literalmente constantes). Cai para estimativa
            # clássica, que é o comportamento honesto — não mascarar.
            centro, cov = np.median(X, axis=0), np.cov(X, rowvar=False)
            cov = np.atleast_2d(cov)

        cov = cov + np.eye(cov.shape[0]) * (self.crista * max(np.trace(cov), 1e-9))
        self.centro_ = centro
        self.prec_ = np.linalg.pinv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        d = np.asarray(X, dtype=float) - self.centro_
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.prec_, d), 0.0))


# ─────────────────────────────────────────────────────────────────────────────
class FlorestaIsolamento:
    """Isolation Forest — número de partições aleatórias para isolar a amostra."""

    def __init__(self, n_estimators: int = 300, contamination: float = 0.01):
        self.modelo = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=SEED,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> "FlorestaIsolamento":
        self.modelo.fit(np.asarray(X, dtype=float))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function: quanto MAIOR, mais normal. Invertemos para que em
        # todos os detectores "maior = mais anômalo".
        return -self.modelo.decision_function(np.asarray(X, dtype=float))


# ─────────────────────────────────────────────────────────────────────────────
class _Rede(nn.Module):
    """16 → 8 → 4 → 8 → 16.

    Rede deliberadamente PEQUENA. Há entre 1.100 e 2.100 amostras de treino por
    ativo e regime; uma rede maior decoraria o conjunto e reconstruiria bem até
    as anomalias, que é o modo de falha clássico de autoencoder para detecção.
    O gargalo de 4 neurônios para 16 features força a rede a usar a redundância
    entre canais (v_rms e a_rms correlacionam 0,98 em regime).
    """

    def __init__(self, n_in: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))


class Autoencoder:
    """Erro de reconstrução como score, agregado por MÉDIA MAIS MÁXIMO.

    Não é o EQM puro. O EQM dilui desvio concentrado: um sensor travado desloca
    UMA feature das 16, cujo erro de reconstrução chega a 132x o normal, mas
    diluído em 16 termos mal move a média — o recall desse modo ficava em 0,01.
    Somar o máximo recupera esse caso sem perder o anterior: a média responde a
    desvios difusos, o máximo a desvios concentrados num único canal.
    Não precisa de pesos porque o resultado é normalizado logo depois.
    """

    def __init__(self, n_in: int, epocas: int = 400, paciencia: int = 40,
                 lote: int = 128, lr: float = 1e-3, weight_decay: float = 1e-5):
        torch.manual_seed(SEED)
        self.n_in = n_in
        self.epocas = epocas
        self.paciencia = paciencia
        self.lote = lote
        self.lr = lr
        self.weight_decay = weight_decay
        self.rede = _Rede(n_in)
        self.epocas_treinadas = 0

    def fit(self, X: np.ndarray) -> "Autoencoder":
        torch.manual_seed(SEED)
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        n_val = max(1, int(0.2 * len(X)))
        # Validação interna CRONOLÓGICA (fim da série), não aleatória: com
        # amostras vizinhas quase idênticas, split aleatório vazaria.
        Xtr, Xval = X[:-n_val], X[-n_val:]
        if len(Xtr) < 2:
            Xtr, Xval = X, X

        opt = torch.optim.Adam(self.rede.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.MSELoss()

        melhor, espera, melhor_estado = float("inf"), 0, None
        for ep in range(self.epocas):
            self.rede.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), self.lote):
                lote = Xtr[perm[i : i + self.lote]]
                opt.zero_grad()
                perda = crit(self.rede(lote), lote)
                perda.backward()
                opt.step()

            self.rede.eval()
            with torch.no_grad():
                vperda = crit(self.rede(Xval), Xval).item()
            if vperda < melhor - 1e-9:
                melhor, espera = vperda, 0
                melhor_estado = {k: v.clone() for k, v in self.rede.state_dict().items()}
            else:
                espera += 1
                if espera >= self.paciencia:
                    break
            self.epocas_treinadas = ep + 1

        if melhor_estado is not None:
            self.rede.load_state_dict(melhor_estado)
        return self

    def erro_por_feature(self, X: np.ndarray) -> np.ndarray:
        """Erro quadrático por amostra e por feature — base da atribuição."""
        self.rede.eval()
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X, dtype=np.float32))
            rec = self.rede(Xt)
            return ((Xt - rec) ** 2).numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        e = self.erro_por_feature(X)
        return e.mean(axis=1) + e.max(axis=1)

    def state_dict(self) -> dict:
        return {k: v.tolist() for k, v in self.rede.state_dict().items()}

    def load_state(self, d: dict) -> "Autoencoder":
        self.rede.load_state_dict({k: torch.tensor(v) for k, v in d.items()})
        self.rede.eval()
        return self


DETECTORES = {
    "mahalanobis": MahalanobisRobusta,
    "isolation_forest": FlorestaIsolamento,
    "autoencoder": Autoencoder,
}
