import json
import logging
from collections import deque
from typing import Any

from fastapi import WebSocket

log = logging.getLogger("ws.manager")

# Quantas leituras acumular antes de rodar inferência ML
# Deve ser >= WINDOW_SIZE do LSTM (60) — padding com zeros causa erro artificial
ML_WINDOW = 60


class ConnectionManager:
    def __init__(self):
        self._clients: dict[int, set[WebSocket]] = {}
        self.sensors:  dict[int, bool] = {}
        # janela deslizante de leituras por componente para inferência
        self._windows: dict[int, deque] = {}
        # modelos carregados uma vez no startup
        self._ml_models: dict[str, Any] | None = None

    def set_ml_models(self, models: dict) -> None:
        self._ml_models = models
        log.info("Modelos ML registrados no ConnectionManager")

    async def connect(self, componente_id: int, ws: WebSocket):
        await ws.accept()
        self._clients.setdefault(componente_id, set()).add(ws)
        self._windows.setdefault(componente_id, deque(maxlen=ML_WINDOW))

    def disconnect(self, componente_id: int, ws: WebSocket):
        self._clients.get(componente_id, set()).discard(ws)

    async def broadcast(self, componente_id: int, payload: dict):
        clients = list(self._clients.get(componente_id, set()))
        for ws in clients:
            try:
                await ws.send_text(json.dumps(payload, default=str))
            except Exception:
                self.disconnect(componente_id, ws)

    def push_leitura(self, componente_id: int, leitura_dict: dict) -> bool:
        """Adiciona leitura à janela. Retorna True quando a janela estiver cheia."""
        window = self._windows.setdefault(componente_id, deque(maxlen=ML_WINDOW))
        window.append(leitura_dict)
        return len(window) >= ML_WINDOW

    def get_window(self, componente_id: int) -> list[dict]:
        return list(self._windows.get(componente_id, []))

    @property
    def ml_ready(self) -> bool:
        return self._ml_models is not None

    @property
    def ml_models(self):
        return self._ml_models


manager = ConnectionManager()
