"""
Índice de erro de reconstrução / score IF → saúde 0-100% (100 = saudável).
"""

import numpy as np


def health_index_from_anomaly(if_score, if_threshold, ae_error, ae_threshold):
    """
    Converte saídas dos detectores de anomalia num índice de saúde 0-100%.
    100% = totalmente saudável, 0% = anomalia máxima.

    if_score: decision_function do Isolation Forest (maior = mais normal)
    ae_error: erro de reconstrução do autoencoder (menor = mais normal)
    """
    if_health = np.clip((if_score - if_threshold) / (abs(if_threshold) + 1e-9), 0, 1)
    ae_health = np.clip(1 - (ae_error / (ae_threshold * 3 + 1e-9)), 0, 1)
    health = (0.4 * if_health + 0.6 * ae_health) * 100
    return float(np.clip(health, 0, 100))
