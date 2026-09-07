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
_CACHE_MTIME: float | None = None


def _mtime_artefatos() -> float:
    """Maior mtime entre os artefatos treinados, para invalidar o cache.

    Sem isso, retreinar exigiria reiniciar a API: o cache de módulo seguraria os
    pesos antigos para sempre e ninguém perceberia que o retreino não surtiu efeito.
    """
    t = 0.0
    for nome in ("modelos.json", "baseline_regime.json", "metricas.json"):
        p = SAVED / nome
        if p.exists():
            t = max(t, p.stat().st_mtime)
    return t


def carregar() -> dict | None:
    """Carrega os artefatos treinados. Devolve None se não houver treino no disco.

    A inferência reconstrói o AUTOENCODER a partir do `state_dict` salvo em
    `modelos.json`, que ocupa poucos KB. Não carrega `detectores.joblib` (23 MB,
    dominado pelas seis florestas de 300 árvores): a Mahalanobis e o Isolation
    Forest existem para a COMPARAÇÃO da bancada, cujo resultado já está gravado
    em `metricas.json`. Em produção só o autoencoder pontua.
    """
    global _CACHE, _CACHE_MTIME
    if not (SAVED / "modelos.json").exists():
        return None

    mtime = _mtime_artefatos()
    if _CACHE is not None and _CACHE_MTIME == mtime:
        return _CACHE

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
    _CACHE_MTIME = mtime
    return _CACHE


# Só estes dois componentes têm modelo treinado, porque só eles aparecem no
# histórico de telemetria. O componente 1 é o motor da FIAP, que é OUTRO motor
# físico: usar a baseline do S1 nele daria um número plausível e errado.
MOTOR_POR_COMPONENTE = {2: "MOTOR-01", 3: "MOTOR-02"}

# Faixas físicas válidas do Metric Contract. Leitura fora disso é falha de
# sensor, não condição do motor, e é descartada antes de pontuar.
FAIXA_VALIDA = {
    "v_rms": (0.0, 50.0),      # mm/s
    "a_rms": (0.0, 20.0),      # g
    "temp_c": (-10.0, 150.0),  # °C
}


def _motor_de_componente(componente_id: int) -> str | None:
    """Motor correspondente ao componente, ou None se não há modelo para ele."""
    return MOTOR_POR_COMPONENTE.get(int(componente_id))


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

    motor = _motor_de_componente(componente_id)
    if motor is None:
        return _sem_modelo(
            f"Componente {componente_id} não tem modelo treinado. Só os componentes "
            f"{sorted(MOTOR_POR_COMPONENTE)} aparecem no histórico de telemetria; "
            "usar a baseline de outro motor daria um número plausível e errado."
        )

    if len(leituras) < 20:
        return _sem_modelo(f"Janela curta ({len(leituras)} leituras; mínimo 20).")

    df = pd.DataFrame(leituras)
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})

    # O banco devolve `timestamptz`, ou seja, timestamps COM fuso; o CSV histórico
    # vem sem fuso. Converter direto com .astype("datetime64[us]") levanta
    # TypeError na entrada com fuso, o que derrubava o endpoint em produção
    # enquanto os testes sobre o CSV passavam. Normalizamos para UTC e removemos
    # o fuso, de modo que os dois caminhos produzam a mesma coisa.
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None).astype("datetime64[us]")

    # Uma leitura sem valor não pode virar zero: zero é "motor parado", que é um
    # estado físico legítimo e mudaria o regime. Descartamos a linha. O mesmo vale
    # para leitura fora da faixa física: é falha de sensor, não condição do motor.
    n_bruto = len(df)
    for c, (lo, hi) in FAIXA_VALIDA.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    df = df.dropna(subset=[c for c in FAIXA_VALIDA if c in df.columns])
    n_descartadas = n_bruto - len(df)
    if len(df) < 20:
        return _sem_modelo(
            f"Janela curta após descartar leituras inválidas ({len(df)}; mínimo 20)."
        )
    df = df.reset_index(drop=True)
    if "a_peak" not in df.columns:
        df["a_peak"] = np.nan
    sem_pico = df["a_peak"].isna().all()
    if sem_pico:
        # Sem o canal de pico, o crest não é observável. Preenchemos com a
        # mediana saudável medida (3,22) para o vetor ficar denso, mas marcamos
        # a limitação — o detector não deve creditar nem debitar rolamento a
        # partir de um crest fabricado.
        df["a_peak"] = df["a_rms"] * 3.22

    df["motor"] = motor
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)

    # O regime da amostra atual usa a classificação de INFERÊNCIA, sem o descarte
    # de cabeça e cauda — aquele descarte é higiene do conjunto de treino e aqui
    # marcaria toda última amostra como transiente.
    regime_atual = regimes.classificar_atual(df)
    df["regime"] = regime_atual
    seg = feat.construir(df)
    chave = f"{motor}|{regime_atual}"

    limitacoes = []
    if n_descartadas:
        limitacoes.append(
            f"{n_descartadas} leitura(s) descartada(s) por estarem fora da faixa física "
            "válida ou sem valor."
        )
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
        # Mantém a MESMA forma dos outros retornos de atribuição. O tipo
        # `AtribuicaoDetalhe` do frontend declara regime, score_normalizado e
        # top_features como obrigatórios; devolver um dict curto aqui faria esses
        # campos chegarem como undefined e o card do 3D quebraria.
        atrib = {
            "atribuido": False,
            "motivo": f"Sem baseline de regime para {chave}.",
            "regime": regime_atual,
            "score_normalizado": round(score, 4),
            "top_features": [],
        }
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
