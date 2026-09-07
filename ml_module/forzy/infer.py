"""
Inferência em produção — pontua a janela mais recente e atribui componentes.

Este é o módulo que a API chama. Ele carrega os 6 modelos treinados, decide o
regime da janela atual, pontua com o autoencoder (o detector escolhido) e devolve
a atribuição por componente, incluindo os segmentos do modelo 3D a destacar.

DUAS GUARDAS FÍSICAS OBRIGATÓRIAS, ambas medidas nos dados reais:

  HEAT SOAK. Depois de uma corrida longa a temperatura da carcaça CONTINUA
  SUBINDO com o motor já parado: 38 °C no fim da corrida chegando a um pico de
  46 °C cerca de 7,9 min depois de desligar. Um detector que tratasse
  "temperatura subindo" como sobreaquecimento alarmaria aqui e estaria errado.

  DERIVA TÉRMICA DO PLATÔ. Durante a corrida a vibração CAI de 6,95 para 6,42
  mm/s enquanto a carcaça aquece. Queda de vibração NUNCA indica degradação, e a
  atribuição só considera desvio positivo.

E o regime TRANSIENTE é pontuado mas NÃO gera alarme, por projeto.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_module.forzy import attribution, features as feat, regimes
from ml_module.forzy.calibration import Normalizador, severidade
from ml_module.forzy.scaling import EscalaRobusta

SAVED = Path(__file__).resolve().parent / "saved"

HEAT_SOAK_S = 900.0      # 15 min após o desligamento em que temperatura alta é normal

_CACHE: dict | None = None


def carregar() -> dict | None:
    """Carrega os artefatos treinados. Devolve None se não houver treino no disco.

    A inferência reconstrói o AUTOENCODER a partir do `state_dict` salvo em
    `modelos.json`, que ocupa poucos KB. Não carrega `detectores.joblib` (23 MB,
    dominado pelas seis florestas de 300 árvores): a Mahalanobis e o Isolation
    Forest existem para a COMPARAÇÃO da bancada, cujo resultado já está gravado
    em `metricas.json`. Em produção só o autoencoder pontua.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not (SAVED / "modelos.json").exists():
        return None

    from ml_module.forzy.detectors import Autoencoder

    with open(SAVED / "modelos.json", encoding="utf-8") as f:
        modelos = json.load(f)
    with open(SAVED / "baseline_regime.json", encoding="utf-8") as f:
        baseline = json.load(f)
    metricas = {}
    if (SAVED / "metricas.json").exists():
        with open(SAVED / "metricas.json", encoding="utf-8") as f:
            metricas = json.load(f)

    n_features = len(modelos.get("features", []))
    redes: dict[str, Autoencoder] = {}
    for chave, entrada in modelos.get("modelos", {}).items():
        estado = entrada.get("detectores", {}).get("autoencoder", {}).get("state_dict")
        if estado:
            redes[chave] = Autoencoder(n_features).load_state(estado)

    _CACHE = {"modelos": modelos, "baseline": baseline,
              "autoencoders": redes, "metricas": metricas}
    return _CACHE


def _motor_de_componente(componente_id: int) -> str:
    """Componente 2 = sensor S1 = MOTOR-01; componente 3 = sensor S2 = MOTOR-02.
    Componente 1 (motor da FIAP) não tem modelo próprio: cai no MOTOR-01."""
    return "MOTOR-02" if int(componente_id) == 3 else "MOTOR-01"


def _sem_modelo(motivo: str) -> dict:
    return {"disponivel": False, "motivo": motivo, "atribuicao": None,
            "segmentos": [], "score_normalizado": None, "severidade": None}


def avaliar(
    leituras: list[dict], componente_id: int, segundos_desde_parada: float | None = None
) -> dict:
    """Pontua a janela de leituras e devolve o diagnóstico com atribuição.

    `leituras` é uma lista de dicts com as chaves `timestamp`, `v_rms`, `a_rms`,
    `temp_c` e opcionalmente `a_peak` (que só existe no arquivo histórico — em
    produção o poller não recebe o canal de pico, então o crest fica degradado e
    isso é reportado em `limitacoes`).
    """
    art = carregar()
    if art is None:
        return _sem_modelo("Modelos não treinados. Rode: python -m ml_module.forzy.train")
    if len(leituras) < 20:
        return _sem_modelo(f"Janela curta ({len(leituras)} leituras; mínimo 20).")

    df = pd.DataFrame(leituras)
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    df["ts"] = pd.to_datetime(df["ts"]).astype("datetime64[us]")
    if "a_peak" not in df.columns:
        df["a_peak"] = np.nan
    sem_pico = df["a_peak"].isna().all()
    if sem_pico:
        # Sem o canal de pico, o crest não é observável. Preenchemos com a
        # mediana saudável medida (3,22) para o vetor ficar denso, mas marcamos
        # a limitação — o detector não deve creditar nem debitar rolamento a
        # partir de um crest fabricado.
        df["a_peak"] = df["a_rms"] * 3.22

    df["motor"] = _motor_de_componente(componente_id)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)

    # O regime da amostra atual usa a classificação de INFERÊNCIA, sem o descarte
    # de cabeça e cauda — aquele descarte é higiene do conjunto de treino e aqui
    # marcaria toda última amostra como transiente.
    regime_atual = regimes.classificar_atual(df)
    df["regime"] = regime_atual
    seg = feat.construir(df)
    motor = df["motor"].iloc[0]
    chave = f"{motor}|{regime_atual}"

    limitacoes = []
    if sem_pico:
        limitacoes.append(
            "Canal a-Peak ausente em produção: o fator de crest foi estimado, não medido. "
            "A evidência de rolamento fica enfraquecida."
        )

    if regime_atual == regimes.PARADO:
        soak = (segundos_desde_parada is not None and segundos_desde_parada < HEAT_SOAK_S)
        return {**_sem_modelo(
            "Motor parado — análise de anomalia não aplicável." +
            (" Temperatura elevada aqui é resfriamento pós-operação (heat soak), não falha."
             if soak else "")),
            "regime": regime_atual, "heat_soak": soak, "limitacoes": limitacoes}

    if chave not in art["modelos"]["modelos"]:
        return {**_sem_modelo(f"Sem modelo treinado para {chave}."),
                "regime": regime_atual, "limitacoes": limitacoes}

    entrada = art["modelos"]["modelos"][chave]
    escala = EscalaRobusta.from_dict(entrada["escala"])
    det = art["autoencoders"].get(chave)
    if det is None:
        return {**_sem_modelo(f"Autoencoder ausente para {chave}."), "regime": regime_atual}
    norm = Normalizador.from_dict(entrada["detectores"]["autoencoder"]["normalizador"])

    x = seg[feat.FEATURES].to_numpy(dtype=float)[-1:]
    score = float(norm(det.score(escala.transform(x)))[0])
    sev = severidade(score)

    # Atribuição por z-score contra o baseline do regime da própria amostra.
    base = art["baseline"]["baseline"].get(chave)
    if base is None:
        atrib = {"atribuido": False, "motivo": f"Sem baseline de regime para {chave}."}
    else:
        atrib = attribution.atribuir(
            x[0], np.asarray(base["mediana"]), np.asarray(base["escala"]),
            score, regime_atual, art["modelos"]["features"],
        )

    # TRANSIENTE é pontuado mas não alarma: durante partida e parada por inércia
    # a vibração varia legitimamente em uma ordem de grandeza.
    alarma = regime_atual == regimes.OPERACAO and score > 1.0
    if regime_atual == regimes.TRANSIENTE:
        limitacoes.append(
            "Regime transiente (partida ou parada): pontuado mas sem alarme, porque a "
            "vibração varia legitimamente em uma ordem de grandeza nessa fase."
        )

    segmentos = atrib.get("segmentos", []) if (alarma and atrib.get("atribuido")) else []

    return {
        "disponivel": True,
        "regime": regime_atual,
        "modelo": chave,
        "score_normalizado": round(score, 4),
        "severidade": sev,
        "alarma": bool(alarma),
        "atribuicao": atrib,
        "segmentos": segmentos,
        "limitacoes": limitacoes,
        "fp_esperado": entrada["detectores"]["autoencoder"].get("fp_teste"),
    }


def metricas_treino() -> dict:
    """Métricas da bancada de injeção, para exibir na UI e para o agente."""
    art = carregar()
    return art["metricas"] if art else {}
