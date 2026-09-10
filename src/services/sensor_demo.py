"""
Modo demonstração — repete o histórico real capturado
(sensor/History_32026-05-19T11-46-10-920.csv) em vez de consultar o hardware
Forzy, que hoje devolve dado zerado (defeito confirmado pelo fabricante).

Espelha a estrutura de forzy_poller.py: mesmo padrão de loop assíncrono e
mesma tabela de destino (leitura_sensor) — só que a fonte é o CSV histórico
em vez de GET /get_s1 / GET /get_s2. Cada leitura gravada aqui é marcada com
origem='demo' (migrations/007_leitura_sensor_origem.sql), para nunca ser
confundida com dado real do hardware.

Ativado via SENSOR_MODE=demo (src/config.py) — nunca roda ao mesmo tempo que
o poller real (ver branch em main.py, dentro do evento de startup).

Controlado por src/routers/demo_control.py (ativar_falha/resetar_falha/status_falha).
"""

import asyncio
import csv
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("forzy.sensor_demo")

CSV_PATH = Path(__file__).resolve().parents[2] / "sensor" / "History_32026-05-19T11-46-10-920.csv"

# Intervalo fixo, não os deltas reais do CSV — os deltas reais têm trechos
# parados de dezenas de minutos, que numa demo só produziriam gráfico flat
# por muito tempo. Loop contínuo comprime as ~4h originais (incluindo as
# transições parado→transiente→operação, que continuam aparecendo) num ciclo
# curto, bom para apresentação.
REPLAY_INTERVAL_SECONDS = 3.0

COMPONENTE_ID_S1 = 2  # Motor WEG W22 - Unidade S1
COMPONENTE_ID_S2 = 3  # Motor WEG W22 - Unidade S2

# Duração total que a falha fica ativa antes de resetar sozinha. Bem maior que
# o EPISODIO_S de ml_module/forzy/injection.py (90s, pensado pra bancada
# offline) de propósito: numa demo ao vivo, uma janela curta quase sempre
# terminava antes do próximo ciclo do scheduler de diagnóstico (300s) chegar a
# ver a falha — o sintoma era "às vezes acende, às vezes não".
EPISODIO_S = 240.0

# Tempo pra rampa alcançar o efeito PLENO (frac=1) — depois disso o efeito
# fica constante até o fim do episódio, em vez de continuar subindo
# linearmente pela duração toda. Sem isso, checar o diagnóstico cedo demais no
# episódio pegava a falha ainda fraca demais pra cruzar o limite.
RAMPA_S = 10.0

# Mesmos pesos de severidade de ml_module/forzy/injection.py (SEVERIDADES).
SEVERIDADES = {"incipiente": 0.35, "moderada": 0.70, "severa": 1.00}

# Mesmos passos de quantização de ml_module/forzy/etl.py (PASSO_QUANT/PASSO_TEMP).
PASSO_QUANT = 0.01
PASSO_TEMP = 1.0

# Estado de falha ativa por componente — mutado por src/routers/demo_control.py.
_FALHA_ATIVA: dict[int, dict] = {}


def ativar_falha(componente_id: int, modo: str, severidade: str) -> None:
    if severidade not in SEVERIDADES:
        raise ValueError(f"severidade inválida: {severidade}")
    _FALHA_ATIVA[componente_id] = {
        "modo": modo,
        "severidade": severidade,
        "inicio": time.monotonic(),
        "congelado": None,  # snapshot usado só pelo modo F4 (sensor travado)
    }
    logger.info("componente=%d — falha de demonstração ativada: modo=%s severidade=%s", componente_id, modo, severidade)


def resetar_falha(componente_id: int) -> None:
    _FALHA_ATIVA.pop(componente_id, None)
    logger.info("componente=%d — falha de demonstração desativada, replay limpo.", componente_id)


def status_falha(componente_id: int) -> dict | None:
    """Estado atual da falha, com `frac` (progresso 0→1 do episódio).
    Reseta sozinho quando o episódio (EPISODIO_S) termina."""
    estado = _FALHA_ATIVA.get(componente_id)
    if not estado:
        return None
    decorrido = time.monotonic() - estado["inicio"]
    if decorrido >= EPISODIO_S:
        resetar_falha(componente_id)
        return None
    return {**estado, "frac": min(1.0, decorrido / RAMPA_S)}


def _quantizar(v_rms: float, a_rms: float, a_peak: float, temp_c: float) -> tuple[float, float, float, float]:
    """Equivalente escalar de `ml_module.forzy.etl.requantizar` (que só aceita
    DataFrame) — mesmos passos de quantização, para o dado injetado ficar na
    mesma grade do sensor real. Ver etl.py:112-127 para a justificativa."""
    q = lambda x: round(x / PASSO_QUANT) * PASSO_QUANT
    return q(v_rms), q(a_rms), q(a_peak), round(temp_c / PASSO_TEMP) * PASSO_TEMP


def _perturbar(v: float, a: float, p: float, t: float, modo: str, frac: float, k: float) -> tuple[float, float, float, float]:
    """Física de cada modo de falha, ESPELHANDO ml_module/forzy/injection.py:171-217
    (mantida manualmente em sincronia — ver lá para a justificativa física de
    cada fórmula e os números medidos que a validam). Essa versão é escalar
    (um ponto por vez, ao vivo); a de injection.py é vetorizada (lote, offline,
    usada pela bancada de injeção de falhas do treino/validação do ML).

    `frac` é o progresso do episódio (0→1) já multiplicado pela severidade
    quando o modo usa rampa; `k` é o peso de severidade puro (só usado por
    modos em degrau, como F5). F4 (sensor travado) não passa por aqui — é
    tratado por congelamento de valor em sensor_demo.py.
    """
    if modo == "F1":  # Desbalanceamento: v-RMS sobe até 14,7 mm/s, crest cai
        alvo = 14.7
        v = v + frac * (alvo - v)
        a = a * (1.0 + 0.25 * frac)
        p = p * (1.0 + 0.10 * frac)
    elif modo == "F2":  # Sobreaquecimento: +22°C, vibração quase inalterada
        t = t + 22.0 * frac
        v = v * (1.0 + 0.03 * frac)
    elif modo == "F3":  # Rolamento: a-Peak dispara (crest 3,2→6,8), v-RMS só +18%
        pulso = 1.0 if random.random() < 0.12 else 0.0
        f = frac * (1.0 + 2.0 * pulso)
        v = v * (1.0 + 0.18 * f)
        a = a * (1.0 + 0.30 * f)
        p = p * (1.0 + 1.125 * f)
    elif modo == "F5":  # Deriva de calibração: ganho errado até +40% — DEGRAU de propósito
        g = 1.0 + 0.40 * k
        v, a, p = v * g, a * g, p * g
    elif modo == "F6":  # Parada não programada: acionamento se perde
        f = k  # degrau — constante durante todo o episódio
        v = v * (1.0 - 0.97 * f)
        a = a * (1.0 - 0.99 * f)
        p = p * (1.0 - 0.99 * f)
    return v, a, p, t


def _aplicar_falha_se_ativa(componente_id: int, dados: dict) -> dict:
    estado = status_falha(componente_id)
    if not estado:
        return dados

    v, a, t = dados["rpm"], dados["vibracao"], dados["temperatura"]
    p = a  # não temos a-Peak separado em produção — ver limitação já documentada

    if estado["modo"] == "F4":  # Sensor travado: congela no valor do início do episódio
        if estado["congelado"] is None:
            _FALHA_ATIVA[componente_id]["congelado"] = (v, a, t)
        v, a, t = _FALHA_ATIVA[componente_id]["congelado"]
        p = a
    else:
        k = SEVERIDADES[estado["severidade"]]
        v, a, p, t = _perturbar(v, a, p, t, estado["modo"], estado["frac"] * k, k)

    v, a, _, t = _quantizar(v, a, p, t)
    return {"rpm": v, "vibracao": a, "temperatura": t}


def _carregar_csv() -> list[dict]:
    linhas: list[dict] = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for _ in range(3):
            next(reader, None)  # 3 linhas de header (tag, nome amigável, tipo)
        for row in reader:
            if len(row) < 9:
                continue
            try:
                linhas.append({
                    "s1": {"rpm": float(row[3]), "vibracao": float(row[4]), "temperatura": float(row[5])},
                    "s2": {"rpm": float(row[6]), "vibracao": float(row[7]), "temperatura": float(row[8])},
                })
            except ValueError:
                continue
    return linhas


def _write_reading(componente_id: int, dados: dict) -> None:
    from src.db.postgres import SessionLocal
    import sqlalchemy as sa

    db = SessionLocal()
    try:
        db.execute(sa.text("""
            INSERT INTO leitura_sensor (componente_id, timestamp, temperatura, rpm, vibracao, origem)
            VALUES (:cid, :ts, :temp, :rpm, :vib, 'demo')
        """), {
            "cid": componente_id, "ts": datetime.now(timezone.utc),
            "temp": dados["temperatura"], "rpm": dados["rpm"], "vib": dados["vibracao"],
        })
        db.commit()
    except Exception as e:
        logger.error("componente=%d — erro ao gravar leitura de demonstração: %s", componente_id, e)
        db.rollback()
    finally:
        db.close()


async def demo_loop() -> None:
    """Loop principal do modo demonstração — substitui poll_loop() quando
    SENSOR_MODE=demo. Nunca deve rodar junto com o poller real (ver main.py)."""
    logger.info(
        "Modo DEMONSTRAÇÃO iniciado — replay de %s a cada %.1fs | S1→componente=%d | S2→componente=%d",
        CSV_PATH.name, REPLAY_INTERVAL_SECONDS, COMPONENTE_ID_S1, COMPONENTE_ID_S2,
    )

    linhas = _carregar_csv()
    if not linhas:
        logger.error("CSV de demonstração vazio ou não encontrado em %s — modo demo não vai gravar nada.", CSV_PATH)
        return

    logger.info("Histórico carregado: %d leituras.", len(linhas))
    index = 0
    while True:
        linha = linhas[index % len(linhas)]
        index += 1

        for componente_id, chave in ((COMPONENTE_ID_S1, "s1"), (COMPONENTE_ID_S2, "s2")):
            dados = _aplicar_falha_se_ativa(componente_id, linha[chave])
            _write_reading(componente_id, dados)

        await asyncio.sleep(REPLAY_INTERVAL_SECONDS)
