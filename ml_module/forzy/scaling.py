"""
Padronização robusta com PISO e CLIPPING.

Subtrai a mediana e divide pelo desvio, mas com dois cuidados obrigatórios.

PISO: dentro de um único regime várias features são quase constantes. O a-RMS com
o motor parado tem desvio de 0,001 g; dividir por ele transforma ruído de
quantização em z-scores de centenas. Sem o piso, a primeira versão produzia
scores da ordem de 26 milhões no regime PARADO.

CLIPPING: impede que uma feature saturada domine o score sozinha.
"""

from __future__ import annotations

import numpy as np

PISO_ESCALA = 0.05    # desvio mínimo = 5% do desvio GLOBAL daquela feature
LIMITE_Z = 10.0       # z-score final clipado


class EscalaRobusta:
    """Mediana/desvio por feature, com piso derivado do desvio global."""

    def __init__(self, piso: float = PISO_ESCALA, limite: float = LIMITE_Z):
        self.piso = piso
        self.limite = limite
        self.mediana_: np.ndarray | None = None
        self.escala_: np.ndarray | None = None

    def fit(self, X: np.ndarray, desvio_global: np.ndarray | None = None) -> "EscalaRobusta":
        X = np.asarray(X, dtype=float)
        self.mediana_ = np.nanmedian(X, axis=0)
        desvio_local = np.nanstd(X, axis=0)
        if desvio_global is None:
            desvio_global = desvio_local
        piso_abs = self.piso * np.asarray(desvio_global, dtype=float)
        # Evita piso zero em feature globalmente constante.
        piso_abs = np.where(piso_abs > 0, piso_abs, 1e-9)
        self.escala_ = np.maximum(desvio_local, piso_abs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mediana_ is None or self.escala_ is None:
            raise RuntimeError("EscalaRobusta não foi ajustada.")
        z = (np.asarray(X, dtype=float) - self.mediana_) / self.escala_
        z = np.nan_to_num(z, nan=0.0, posinf=self.limite, neginf=-self.limite)
        return np.clip(z, -self.limite, self.limite)

    def fit_transform(self, X: np.ndarray, desvio_global: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, desvio_global).transform(X)

    def to_dict(self) -> dict:
        return {
            "mediana": self.mediana_.tolist(),
            "escala": self.escala_.tolist(),
            "piso": self.piso,
            "limite": self.limite,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EscalaRobusta":
        obj = cls(piso=d.get("piso", PISO_ESCALA), limite=d.get("limite", LIMITE_Z))
        obj.mediana_ = np.asarray(d["mediana"], dtype=float)
        obj.escala_ = np.asarray(d["escala"], dtype=float)
        return obj
