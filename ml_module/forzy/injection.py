"""
Bancada de injeção de falhas.

Sem rótulos não existe recall, F1 nem PR-AUC. A saída é perturbar o dado de TESTE
saudável com degradações fisicamente calibradas e medir se o detector — treinado
apenas em dado saudável e que nunca viu nenhuma injeção — as separa do normal.

ALCANCE: isso prova que o detector reconhece desvios com a assinatura física
esperada e quantifica a margem entre normal e anômalo. NÃO prova que detectaria
falha real, porque uma falha de verdade tem componentes não modelados (espectro,
modulação, interação com a carga). As métricas são um LIMITE SUPERIOR OTIMISTA.

QUATRO CUIDADOS METODOLÓGICOS OBRIGATÓRIOS:
  1. Série CONTÍNUA, não janelas soltas. A falha entra no meio de uma corrida em
     andamento e a série é mantida inteira. Recortar janelas isoladas faria o
     detector de regime marcar as bordas de cada recorte como transiente, criando
     artefato do recorte e não do equipamento.
  2. Perturbar os CANAIS BRUTOS e recalcular as 16 features do zero. Alterar a
     feature diretamente cria incoerência entre feature e sinal que qualquer
     detector acha trivialmente.
  3. REQUANTIZAR depois de injetar. Sem isso o detector acusa as falhas pela
     perda de quantização em vez da física — corrigir derrubou a PR-AUC macro de
     0,891 para 0,838, ou seja, 0,891 era ilusão.
  4. ZONA CINZA. Nos 60 s seguintes ao fim de cada episódio as janelas móveis
     ainda carregam amostras contaminadas. Esses pontos são DESCARTADOS das
     métricas, não contados como falso positivo.

Amplitudes calibradas por ISO 10816-3/20816-3 (Classe I, abaixo de 15 kW), pela
dinâmica térmica medida (0,38 °C/min sob carga) e pelo crest medido (3,22 ± 0,82).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_module.forzy.etl import requantizar

SEED = 42
EPISODIO_S = 90.0        # duração de cada episódio de falha
PERIODO_S = 180.0        # um episódio a cada 180 s
N_REPETICOES = 3
ZONA_CINZA_S = 60.0      # descartado das métricas após o fim do episódio

SEVERIDADES = {"incipiente": 0.35, "moderada": 0.70, "severa": 1.00}

MODOS = ["F1", "F2", "F3", "F4", "F5", "F6"]

DESCRICAO = {
    "F1": ("Desbalanceamento", "v-RMS até 14,7 mm/s; a-RMS sobe pouco; crest cai", "rampa"),
    "F2": ("Sobreaquecimento", "temperatura +22 °C; vibração quase inalterada", "rampa"),
    "F3": ("Rolamento", "a-Peak dispara, crest de 3,2 a 6,8; v-RMS só +18%", "rampa com pulsos"),
    "F4": ("Sensor travado", "canais congelam em valores nominais", "degrau"),
    "F5": ("Deriva de calibração", "ganho errado até +40%, valores plausíveis", "degrau"),
    "F6": ("Parada não programada", "acionamento se perde, valores de motor desligado", "degrau"),
}


def _segundos(ts: np.ndarray) -> np.ndarray:
    return (ts - ts[0]) / np.timedelta64(1, "s")


def _blocos(mascara: np.ndarray) -> list[tuple[int, int]]:
    """Índices [ini, fim) de cada trecho contíguo True."""
    if not mascara.any():
        return []
    d = np.diff(mascara.astype(np.int8))
    inis = list(np.flatnonzero(d == 1) + 1)
    fins = list(np.flatnonzero(d == -1) + 1)
    if mascara[0]:
        inis.insert(0, 0)
    if mascara[-1]:
        fins.append(len(mascara))
    return list(zip(inis, fins))


def _episodios(
    seg: np.ndarray, operando: np.ndarray, deslocamento: float = 0.0
) -> list[tuple[float, float]]:
    """Janelas [t0, t1) dos episódios, ANCORADAS nos trechos de operação.

    A primeira versão posicionava os episódios a partir do início da série, o que
    não funciona neste dataset: a operação em regime só acontece em dois trechos,
    a cerca de 42 min e 2 h do início. Todos os episódios caíam com o motor
    parado e a bancada não gerava nenhuma amostra positiva.

    Uma falha de rolamento não se manifesta com o eixo parado, então os episódios
    têm que viver dentro da corrida — que é também o cuidado nº 1: a falha entra
    no MEIO de uma corrida em andamento e a série é mantida inteira.
    """
    eps: list[tuple[float, float]] = []
    for ini, fim in _blocos(operando):
        t_ini, t_fim = seg[ini], seg[fim - 1]
        dur = t_fim - t_ini
        if dur < EPISODIO_S:
            continue
        # Começa deslocado dentro do trecho; o módulo mantém o episódio dentro
        # mesmo quando o deslocamento é maior que a folga disponível.
        folga = max(dur - EPISODIO_S, 0.0)
        t = t_ini + (deslocamento % (folga + 1e-9)) if folga > 0 else t_ini
        while t + EPISODIO_S <= t_fim + 1e-9 and len(eps) < 64:
            eps.append((t, t + EPISODIO_S))
            t += PERIODO_S
    return eps


def _perfil(seg_ep: np.ndarray, t0: float, t1: float, tipo: str) -> np.ndarray:
    """Fator temporal de 0 a 1 dentro do episódio."""
    frac = np.clip((seg_ep - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
    if tipo == "degrau":
        return np.ones_like(frac)
    if tipo == "rampa":
        return frac
    if tipo == "rampa_pulsos":
        # rampa com pulsos curtos de impacto, típico de defeito de pista
        rng = np.random.default_rng(SEED)
        pulsos = (rng.random(len(frac)) < 0.12).astype(float)
        return frac * (1.0 + 2.0 * pulsos)
    return frac


def injetar(
    df_motor: pd.DataFrame,
    modo: str,
    severidade: str,
    deslocamento: float = 0.0,
    elegivel: np.ndarray | None = None,
) -> pd.DataFrame:
    """Aplica um modo de falha nos CANAIS BRUTOS de uma cópia da série.

    Devolve o dataframe com os canais perturbados, mais as colunas `injetado`
    (positivo verdadeiro) e `zona_cinza` (descartar das métricas).
    As features NÃO são calculadas aqui — quem chama deve recalculá-las do zero.

    `elegivel` é a máscara de posições onde o episódio PODE cair. Quem avalia
    deve passar a porção de TESTE do regime de operação: sem isso os episódios
    caem no trecho de treino, a avaliação não encontra nenhum positivo e a
    bancada sai vazia. Omitido, usa todo o regime de operação.
    """
    out = df_motor.reset_index(drop=True).copy()
    seg = _segundos(out["ts"].to_numpy())
    k = SEVERIDADES[severidade]

    v = out["v_rms"].to_numpy(dtype=float).copy()
    a = out["a_rms"].to_numpy(dtype=float).copy()
    p = out["a_peak"].to_numpy(dtype=float).copy()
    t = out["temp_c"].to_numpy(dtype=float).copy()

    injetado = np.zeros(len(out), dtype=bool)
    cinza = np.zeros(len(out), dtype=bool)

    # Só faz sentido injetar onde o motor está de fato operando: uma falha de
    # rolamento não se manifesta com o eixo parado.
    if elegivel is not None:
        operando = np.asarray(elegivel, dtype=bool)
    elif "regime" in out.columns:
        operando = out["regime"].to_numpy() == "OPERACAO"
    else:
        operando = np.ones(len(out), dtype=bool)

    eps = _episodios(seg, operando, deslocamento)[:N_REPETICOES]
    for t0, t1 in eps:
        dentro = (seg >= t0) & (seg < t1) & operando
        if not dentro.any():
            continue
        idx = np.flatnonzero(dentro)
        injetado[idx] = True
        pos = (seg >= t1) & (seg < t1 + ZONA_CINZA_S)
        cinza[pos] = True

        if modo == "F1":
            # Desbalanceamento: v-RMS até 14,7 mm/s (zona D da ISO), a-RMS sobe
            # pouco (energia em 1x rotação, não em alta frequência), crest cai.
            f = _perfil(seg[idx], t0, t1, "rampa") * k
            alvo = 14.7
            v[idx] = v[idx] + f * (alvo - v[idx])
            a[idx] = a[idx] * (1.0 + 0.25 * f)
            p[idx] = p[idx] * (1.0 + 0.10 * f)     # pico sobe menos que o RMS -> crest cai

        elif modo == "F2":
            # Sobreaquecimento: +22 °C, vibração quase inalterada.
            f = _perfil(seg[idx], t0, t1, "rampa") * k
            t[idx] = t[idx] + 22.0 * f
            v[idx] = v[idx] * (1.0 + 0.03 * f)

        elif modo == "F3":
            # Rolamento: a-Peak dispara levando o crest de 3,2 a 6,8; v-RMS só +18%.
            f = _perfil(seg[idx], t0, t1, "rampa_pulsos") * k
            v[idx] = v[idx] * (1.0 + 0.18 * f)
            a[idx] = a[idx] * (1.0 + 0.30 * f)
            p[idx] = p[idx] * (1.0 + 1.125 * f)    # 3,2 -> 6,8 ~= x2,125

        elif modo == "F4":
            # Sensor travado: canais congelam em valores NOMINAIS. Continuam
            # dentro da faixa válida, então só frac_repetida denuncia.
            i0 = idx[0]
            v[idx], a[idx], p[idx], t[idx] = v[i0], a[i0], p[i0], t[i0]

        elif modo == "F5":
            # Deriva de calibração: ganho errado até +40%, valores plausíveis.
            # DEGRAU de propósito: a primeira versão usava rampa de 90 s para uma
            # deriva que na realidade leva semanas, e o detector passava a acusar
            # F5 pela frac_repetida em vez da magnitude — acertava pelo motivo
            # errado. Como degrau, é o que um sensor já derivado parece numa
            # janela curta. A PR-AUC caiu de 0,838 para 0,822 ao corrigir.
            g = 1.0 + 0.40 * k
            v[idx] *= g
            a[idx] *= g
            p[idx] *= g

        elif modo == "F6":
            # Parada não programada: o acionamento se perde. Em k=1 vai a valores
            # de motor desligado; em k intermediário cai parcialmente.
            f = _perfil(seg[idx], t0, t1, "degrau") * k
            v[idx] = v[idx] * (1.0 - 0.97 * f)
            a[idx] = a[idx] * (1.0 - 0.99 * f)
            p[idx] = p[idx] * (1.0 - 0.99 * f)

    out["v_rms"], out["a_rms"], out["a_peak"], out["temp_c"] = v, a, p, t
    out = requantizar(out)                      # cuidado 3
    out["injetado"] = injetado
    out["zona_cinza"] = cinza & ~injetado       # cuidado 4
    return out


def cenarios() -> list[tuple[str, str]]:
    """As 18 células de avaliação: 6 modos x 3 severidades."""
    return [(m, s) for m in MODOS for s in SEVERIDADES]
