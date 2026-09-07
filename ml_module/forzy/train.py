"""
Treino dos 6 modelos: 2 motores x 3 regimes. NÃO um modelo global.

POR REGIME, porque as distribuições não se sobrepõem e um modelo único aceitaria
tudo que existe entre elas, que é justamente onde vivem as falhas de perda de
acionamento.

POR ATIVO, porque as assinaturas diferem: o MOTOR-02 opera 4,39 °C mais quente e
vibra 0,182 mm/s a mais que o MOTOR-01 em regime (medido, p < 1e-10).

O regime TRANSIENTE recebe modelo e é pontuado, mas NÃO gera alarme: durante
partida e parada por inércia a vibração varia legitimamente em uma ordem de
grandeza. Modelá-lo mesmo assim é necessário porque uma perda parcial de
acionamento derruba o v-RMS exatamente para essa faixa.

DIVISÃO TREINO/TESTE: corte cronológico 70/30 aplicado DENTRO de cada bloco
contíguo de regime, não globalmente. Há apenas dois trechos de regime por ativo;
um corte global mandaria todo o final da corrida longa para o teste, justamente o
trecho em que o motor já aqueceu e o v-RMS assentou de 7,0 para 6,4 mm/s. O
modelo nunca teria visto essa condição térmica e marcaria como anomalia uma
deriva perfeitamente normal. Efeito medido do corte global: o falso positivo do
autoencoder sobe de 0,55% para 24,2%.

Uso:
    python -m ml_module.forzy.train
"""

from __future__ import annotations

import json
import sys
import time
import functools
print = functools.partial(print, flush=True)
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.forzy import etl, features as feat, injection, regimes
from ml_module.forzy.calibration import calibrar, severidade
from ml_module.forzy.detectors import Autoencoder, FlorestaIsolamento, MahalanobisRobusta
from ml_module.forzy.scaling import EscalaRobusta

CSV = PROJECT_ROOT / "sensor" / "History_32026-05-19T11-46-10-920.csv"
XLSX = PROJECT_ROOT.parent / "forzy_data.xlsx"
SAIDA = PROJECT_ROOT / "ml_module" / "forzy" / "saved"

FRACAO_TREINO = 0.70
SEED = 42


def _split_por_bloco(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras de treino e teste, com corte 70/30 DENTRO de cada bloco contíguo."""
    treino = np.zeros(len(df), dtype=bool)
    idx = np.arange(len(df))
    blocos = df["bloco"].to_numpy() if "bloco" in df.columns else np.zeros(len(df), int)
    for b in np.unique(blocos):
        sel = idx[blocos == b]
        if sel.size == 0:
            continue
        corte = int(len(sel) * FRACAO_TREINO)
        treino[sel[:corte]] = True
    return treino, ~treino


def _detector_factories(n_in: int) -> dict:
    return {
        "mahalanobis": lambda: MahalanobisRobusta(),
        "isolation_forest": lambda: FlorestaIsolamento(),
        "autoencoder": lambda: Autoencoder(n_in),
    }


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _recall_em_fp(y: np.ndarray, s: np.ndarray, fp_alvo: float) -> float:
    """Recall com o limiar ajustado para o falso positivo alvo.

    É a ÚNICA comparação justa entre modelos: comparar pelo limiar nominal de
    cada um não informa nada, porque um modelo pode parecer melhor apenas por
    alarmar mais.
    """
    neg, pos = s[y == 0], s[y == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    limiar = np.percentile(neg, 100 * (1 - fp_alvo))
    return float((pos > limiar).mean())


def treinar() -> dict:
    t_ini = time.time()
    SAIDA.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("FORZY — TREINO DOS 6 MODELOS (2 motores x 3 regimes)")
    print("=" * 74)

    # ── 1. ETL ────────────────────────────────────────────────────────────
    print("\n[1/6] ETL — decodificando os 4 canais do PDI")
    longo = etl.carregar_csv(CSV)
    print(f"      {len(longo)} leituras em formato longo "
          f"({longo['motor'].nunique()} motores)")

    # ── 2. Regimes ────────────────────────────────────────────────────────
    print("\n[2/6] Segmentando os 3 regimes")
    dados = regimes.segmentar_todos(longo)
    res = regimes.resumo(dados)
    print(res.to_string())

    # ── 3. Features ───────────────────────────────────────────────────────
    print("\n[3/6] Construindo as 16 features (janelas temporais)")
    dados = feat.construir_todos(dados)
    print(f"      {len(feat.FEATURES)} features x {len(dados)} amostras")

    # Desvio GLOBAL por feature — base do piso de escala.
    desvio_global = dados[feat.FEATURES].to_numpy(dtype=float).std(axis=0)

    # ── 4. Treino por ativo e regime ──────────────────────────────────────
    print("\n[4/6] Treinando 6 modelos")
    artefatos: dict = {"features": feat.FEATURES, "modelos": {}}
    resumo_treino = []

    for motor in sorted(dados["motor"].unique()):
        for regime in (regimes.PARADO, regimes.TRANSIENTE, regimes.OPERACAO):
            sub = dados[(dados["motor"] == motor) & (dados["regime"] == regime)].reset_index(drop=True)
            if len(sub) < 40:
                print(f"      {motor}/{regime}: {len(sub)} amostras — insuficiente, pulado")
                continue

            m_tr, m_te = _split_por_bloco(sub)
            X_all = sub[feat.FEATURES].to_numpy(dtype=float)

            escala = EscalaRobusta().fit(X_all[m_tr], desvio_global)
            Z_tr, Z_te = escala.transform(X_all[m_tr]), escala.transform(X_all[m_te])

            chave = f"{motor}|{regime}"
            fabricas = _detector_factories(len(feat.FEATURES))
            entrada = {"escala": escala.to_dict(), "n_treino": int(m_tr.sum()),
                       "n_teste": int(m_te.sum()), "detectores": {}}

            for nome, fab in fabricas.items():
                norm, _ = calibrar(Z_tr, fab)
                det = fab().fit(Z_tr)
                s_te = norm(det.score(Z_te)) if m_te.sum() else np.array([])
                fp = float((s_te > 1.0).mean()) if s_te.size else float("nan")

                d = {"normalizador": norm.to_dict(), "fp_teste": fp}
                if nome == "autoencoder":
                    d["state_dict"] = det.state_dict()
                    d["epocas"] = det.epocas_treinadas
                entrada["detectores"][nome] = d
                entrada.setdefault("_objs", {})[nome] = det

            artefatos["modelos"][chave] = entrada
            ae = entrada["detectores"]["autoencoder"]
            resumo_treino.append({
                "modelo": chave, "treino": int(m_tr.sum()), "teste": int(m_te.sum()),
                "epocas_ae": ae["epocas"], "fp_ae": round(ae["fp_teste"], 4),
            })
            print(f"      {chave:26} treino={m_tr.sum():5} teste={m_te.sum():5} "
                  f"épocas={ae['epocas']:3}  FP={ae['fp_teste']*100:5.2f}%")

    # ── 5. Bancada de injeção de falhas ───────────────────────────────────
    print("\n[5/6] Bancada de injeção — 6 modos x 3 severidades")
    metricas = avaliar_bancada(dados, artefatos, desvio_global)

    # ── 6. Persistência ───────────────────────────────────────────────────
    print("\n[6/6] Salvando artefatos")
    persistir(artefatos, metricas, dados, desvio_global)

    print(f"\nTempo total: {time.time() - t_ini:.1f}s")
    return metricas


def avaliar_bancada(dados: pd.DataFrame, artefatos: dict, desvio_global: np.ndarray) -> dict:
    """Avalia os três detectores contra as 18 células de modo x severidade.

    PARTIÇÃO DE AVALIAÇÃO: cada cenário é uma cópia integral da série de teste,
    então as amostras saudáveis se repetiriam dezenas de vezes. Usamos negativos
    SÓ do cenário de controle e positivos SÓ das amostras injetadas — sem isso os
    positivos viram 85% do pool e a PR-AUC de um classificador aleatório já vale
    0,85.
    """
    por_detector: dict[str, dict[str, list]] = {
        n: {"pr": [], "roc": [], "recall": []} for n in ("mahalanobis", "isolation_forest", "autoencoder")
    }
    celulas: list[dict] = []
    fp_controle: dict[str, list] = {n: [] for n in por_detector}

    for motor in sorted(dados["motor"].unique()):
        chave = f"{motor}|{regimes.OPERACAO}"
        if chave not in artefatos["modelos"]:
            continue
        entrada = artefatos["modelos"][chave]
        escala = EscalaRobusta.from_dict(entrada["escala"])
        objs = entrada["_objs"]

        base_motor = dados[dados["motor"] == motor].reset_index(drop=True)
        sub_op = base_motor[base_motor["regime"] == regimes.OPERACAO]
        m_tr, _ = _split_por_bloco(sub_op.reset_index(drop=True))
        # A bancada perturba a série INTEIRA (cuidado 1), mas só as amostras de
        # OPERACAO fora do treino contam como avaliação.
        idx_teste_op = sub_op.index.to_numpy()[~m_tr]
        # Os episódios têm que cair DENTRO dessa mesma partição de teste, senão
        # perturbamos o trecho de treino e a avaliação não encontra positivo.
        elegivel = np.zeros(len(base_motor), dtype=bool)
        elegivel[idx_teste_op] = True

        # Controle: série saudável, sem injeção.
        ctrl = feat.construir(base_motor)
        Xc = ctrl.loc[idx_teste_op, feat.FEATURES].to_numpy(dtype=float)
        Zc = escala.transform(Xc)
        neg = {}
        for nome, det in objs.items():
            from ml_module.forzy.calibration import Normalizador
            norm = Normalizador.from_dict(entrada["detectores"][nome]["normalizador"])
            neg[nome] = norm(det.score(Zc))
            fp_controle[nome].append(float((neg[nome] > 1.0).mean()))

        from ml_module.forzy.calibration import Normalizador
        normalizadores = {
            n: Normalizador.from_dict(entrada["detectores"][n]["normalizador"])
            for n in por_detector
        }

        for modo, sev in injection.cenarios():
            # A perturbação e o recálculo das features NÃO dependem do detector.
            # Fazer uma vez por (modo, severidade, repetição) e pontuar os três
            # em cima do mesmo X economiza 3x o trabalho, que aqui é o gargalo.
            X_pos = []
            for rep in range(injection.N_REPETICOES):
                pert = injection.injetar(base_motor, modo, sev,
                                         deslocamento=rep * 37.0, elegivel=elegivel)
                pert = feat.construir(pert)
                sel = (pert.index.isin(idx_teste_op)
                       & pert["injetado"].to_numpy()
                       & ~pert["zona_cinza"].to_numpy())
                if sel.any():
                    X_pos.append(escala.transform(
                        pert.loc[sel, feat.FEATURES].to_numpy(dtype=float)))
            if not X_pos:
                continue
            Zp = np.vstack(X_pos)

            for nome in por_detector:
                pos = normalizadores[nome](objs[nome].score(Zp))
                y = np.concatenate([np.zeros(len(neg[nome])), np.ones(len(pos))])
                s = np.concatenate([neg[nome], pos])
                pr, roc = _pr_auc(y, s), _roc_auc(y, s)
                rec = _recall_em_fp(y, s, 0.01)
                por_detector[nome]["pr"].append(pr)
                por_detector[nome]["roc"].append(roc)
                por_detector[nome]["recall"].append(rec)
                if nome == "autoencoder":
                    celulas.append({"motor": motor, "modo": modo, "severidade": sev,
                                    "pr_auc": round(pr, 4), "roc_auc": round(roc, 4),
                                    "recall_fp1": round(rec, 4)})
            print(f"      {motor} {modo}/{sev:<11} "
                  f"PR-AUC(AE)={celulas[-1]['pr_auc']:.4f}", flush=True)

    resultado = {"por_detector": {}, "celulas": celulas}
    for nome, v in por_detector.items():
        resultado["por_detector"][nome] = {
            "pr_auc_macro": round(float(np.nanmean(v["pr"])), 4) if v["pr"] else None,
            "roc_auc_macro": round(float(np.nanmean(v["roc"])), 4) if v["roc"] else None,
            "recall_macro_fp1": round(float(np.nanmean(v["recall"])), 4) if v["recall"] else None,
            "fp_controle": round(float(np.nanmean(fp_controle[nome])), 4) if fp_controle[nome] else None,
        }
    return resultado


def persistir(artefatos: dict, metricas: dict, dados: pd.DataFrame, desvio_global: np.ndarray) -> None:
    """Salva os artefatos que a inferência precisa, sem os objetos vivos."""
    import joblib

    objs = {k: v.pop("_objs") for k, v in artefatos["modelos"].items() if "_objs" in v}
    joblib.dump(objs, SAIDA / "detectores.joblib")

    # Baseline por regime para a atribuição por z-score (determinístico).
    baseline = {}
    for motor in sorted(dados["motor"].unique()):
        for regime in (regimes.PARADO, regimes.TRANSIENTE, regimes.OPERACAO):
            sub = dados[(dados["motor"] == motor) & (dados["regime"] == regime)]
            if len(sub) < 10:
                continue
            X = sub[feat.FEATURES].to_numpy(dtype=float)
            baseline[f"{motor}|{regime}"] = {
                "mediana": np.nanmedian(X, axis=0).tolist(),
                "escala": np.maximum(np.nanstd(X, axis=0), 0.05 * desvio_global).tolist(),
                "n": int(len(sub)),
            }

    with open(SAIDA / "modelos.json", "w", encoding="utf-8") as f:
        json.dump(artefatos, f, ensure_ascii=False, indent=2)
    with open(SAIDA / "baseline_regime.json", "w", encoding="utf-8") as f:
        json.dump({"features": feat.FEATURES, "baseline": baseline,
                   "desvio_global": desvio_global.tolist()}, f, ensure_ascii=False, indent=2)
    with open(SAIDA / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"      -> {SAIDA}")


if __name__ == "__main__":
    m = treinar()
    print("\n" + "=" * 74)
    print("COMPARAÇÃO DOS TRÊS DETECTORES")
    print("=" * 74)
    print(f"{'modelo':<20} {'PR-AUC':>9} {'ROC-AUC':>9} {'Recall@FP1%':>12} {'FP controle':>12}")
    for nome, v in m["por_detector"].items():
        print(f"{nome:<20} {str(v['pr_auc_macro']):>9} {str(v['roc_auc_macro']):>9} "
              f"{str(v['recall_macro_fp1']):>12} {str(v['fp_controle']):>12}")
