import json
import logging
from collections import deque
from typing import Any

from fastapi import WebSocket

log = logging.getLogger("ws.manager")

# Quantas leituras acumular antes de rodar inferência ML.
# 500 registros a 1s/leitura ≈ 8 minutos — cobre as 3 janelas rolling do modelo
# (50, 200, 500) com dados representativos, idêntico ao treino.
# Após análise a janela é zerada (não deslizante).
ML_WINDOW = 500


class ConnectionManager:
    def __init__(self):
        self._clients: dict[int, set[WebSocket]] = {}
        self.sensors:  dict[int, bool] = {}
        # buffer de leituras por componente — resetado após cada inferência
        self._windows: dict[int, list] = {}
        # modelos carregados uma vez no startup
        self._ml_models: dict[str, Any] | None = None

    def set_ml_models(self, models: dict) -> None:
        self._ml_models = models
        log.info("Modelos ML registrados no ConnectionManager")

    async def connect(self, componente_id: int, ws: WebSocket):
        await ws.accept()
        self._clients.setdefault(componente_id, set()).add(ws)
        self._windows.setdefault(componente_id, [])

    def disconnect(self, componente_id: int, ws: WebSocket):
        self._clients.get(componente_id, set()).discard(ws)

    async def broadcast(self, componente_id: int, payload: dict):
        clients = list(self._clients.get(componente_id, set()))
        text = json.dumps(payload, default=str)
        for ws in clients:
            try:
                await ws.send_text(text)
            except Exception as e:
                log.debug("componente=%d — cliente removido do broadcast: %s", componente_id, e)
                self.disconnect(componente_id, ws)

    def push_leitura(self, componente_id: int, leitura_dict: dict) -> bool:
        """Acumula leitura apenas se motor operando (rpm > 0.3 mm/s).
        Retorna True quando atingir ML_WINDOW (janela pronta para análise)."""
        rpm = leitura_dict.get("rpm") or 0.0
        if float(rpm) <= 0.3:
            return False
        window = self._windows.setdefault(componente_id, [])
        window.append(leitura_dict)
        return len(window) >= ML_WINDOW

    def get_window(self, componente_id: int) -> list[dict]:
        return list(self._windows.get(componente_id, []))

    def reset_window(self, componente_id: int) -> None:
        """Zera o buffer após análise ML."""
        self._windows[componente_id] = []

    @property
    def ml_ready(self) -> bool:
        return self._ml_models is not None

    @property
    def ml_models(self):
        return self._ml_models


manager = ConnectionManager()
