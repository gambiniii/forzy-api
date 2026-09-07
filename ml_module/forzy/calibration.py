"""
Calibração do limiar e normalização do score.

Score normalizado como (bruto − p50) / (p_limiar − p50), de modo que 1,0 seja
sempre o ponto de operação, qualquer que seja o detector. Isso torna os três
comparáveis e dá um número interpretável.

Faixas de severidade: NORMAL < 1,0 | ATENÇÃO até 2,0 | ALERTA até 4,0 | CRÍTICO acima.

ESTIMAR OS PERCENTIS FOI A PARTE MAIS DIFÍCIL. Duas abordagens naturais falham,
ambas medidas neste conjunto:

  • Separação cronológica simples (últimos 20% do treino): em OPERACAO isso cai
    numa fatia de 2 min 42 s de uma única corrida, cujo desvio de v-RMS é 0,052
    mm/s contra 0,741 do restante — 14x mais estreita. O limiar sai altíssimo e o
    detector deixa de alarmar (recall macro de 0,12).
  • Separação aleatória: os pontos de calibração ficam intercalados no tempo com
    os de ajuste, praticamente idênticos a eles. O modelo os reconstrói bem
    demais, o limiar sai baixo e o falso positivo real chega a 28%.

SOLUÇÃO: validação cruzada com 5 dobras CONTÍGUAS no tempo. Cada dobra é pontuada
por um modelo ajustado sem ela, e os scores fora-da-dobra são reunidos. O
resultado é uma amostra que é ao mesmo tempo fora da amostra e representativa de
todo o período.

Também foi testado o percentil 95 como ponto de operação: entrega 13,9% de falso
positivo real contra 5% nominal, fator de 2,8. Não se sustenta porque a
distribuição fora-da-dobra tem cauda pesada mas corpo estreito, e o p95 cai
justamente no corpo. Por isso o padrão é p99.
"""

from __future__ import annotations

import numpy as np

N_DOBRAS = 5
PERCENTIL_OPERACAO = 99.0     # 1% de falso positivo nominal

FAIXAS = [(1.0, "NORMAL"), (2.0, "ATENCAO"), (4.0, "ALERTA")]


def scores_fora_da_dobra(
    X: np.ndarray, construir_detector, n_dobras: int = N_DOBRAS
) -> np.ndarray:
    """Pontua cada dobra contígua com um detector ajustado SEM ela.

    `construir_detector` é uma fábrica sem argumentos que devolve um objeto com
    `.fit(X)` e `.score(X)`.
    """
    n = len(X)
    if n < n_dobras * 4:
        # Amostra pequena demais para dobrar: devolve score in-sample, mas quem
        # chama precisa saber que isso é otimista.
        det = construir_detector().fit(X)
        return det.score(X)

    bordas = np.linspace(0, n, n_dobras + 1).astype(int)
    fora = np.empty(n)
    for k in range(n_dobras):
        ini, fim = bordas[k], bordas[k + 1]
        treino = np.concatenate([X[:ini], X[fim:]])
        if len(treino) < 4:
            fora[ini:fim] = np.nan
            continue
        det = construir_detector().fit(treino)
        fora[ini:fim] = det.score(X[ini:fim])
    return fora


class Normalizador:
    """Converte score bruto em score normalizado com 1,0 no ponto de operação."""

    def __init__(self, p50: float, p_limiar: float):
        self.p50 = float(p50)
        self.p_limiar = float(p_limiar)
        # Guarda contra denominador degenerado (regime PARADO, em que várias
        # features são constantes e a distribuição de score colapsa).
        self.denom = max(self.p_limiar - self.p50, 1e-9)

    def __call__(self, bruto: np.ndarray) -> np.ndarray:
        return (np.asarray(bruto, dtype=float) - self.p50) / self.denom

    def to_dict(self) -> dict:
        return {"p50": self.p50, "p_limiar": self.p_limiar}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizador":
        return cls(d["p50"], d["p_limiar"])


def calibrar(
    X: np.ndarray, construir_detector, percentil: float = PERCENTIL_OPERACAO
) -> tuple[Normalizador, np.ndarray]:
    """Devolve (normalizador, scores_fora_da_dobra)."""
    fora = scores_fora_da_dobra(X, construir_detector)
    validos = fora[~np.isnan(fora)]
    if validos.size == 0:
        return Normalizador(0.0, 1.0), fora
    return Normalizador(np.percentile(validos, 50), np.percentile(validos, percentil)), fora


def severidade(score_norm: float) -> str:
    for limite, nome in FAIXAS:
        if score_norm < limite:
            return nome
    return "CRITICO"
