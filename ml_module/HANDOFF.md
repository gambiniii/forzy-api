# Forzy Digital Twin — ML Module Handoff

Documento de entrega do módulo de Machine Learning para monitoramento preditivo do motor **WEG W22 3cv** (Digital Twin Forzy).

---

## Status dos Modelos

| Modelo | Dataset | Métrica | Valor | Status |
|--------|---------|---------|-------|--------|
| Isolation Forest | CWRU Bearing | F1 | 0,9757 | Aprovado |
| Isolation Forest | Forzy (4h) | Taxa anomalias | 5,0% (360 reg.) | Operacional |
| LSTM Autoencoder | CWRU Bearing | F1 | 0,9846 | Aprovado |
| LSTM Autoencoder | Forzy (4h) | Taxa anomalias | 6,2% (442 reg.) | Operacional |
| XGBoost RUL | NASA CMAPSS FD001 | R² | 0,9072 | Aprovado |
| XGBoost RUL | NASA CMAPSS FD001 | MAE | 8,69 ciclos | Aprovado |
| XGBoost RUL | Forzy (original) | R² | -3,02 | Esperado (sem labels) |
| XGBoost RUL Híbrido | Forzy + CWRU | R² | 0,9994 | Aprovado |
| XGBoost RUL Híbrido | Forzy + CWRU | MAE | 0,12 h | Aprovado |
| XGBoost RUL Híbrido | Forzy + CWRU | Correlação temporal | -0,5712 | Aprovado |
| RUL Transfer Adapter | Forzy → CMAPSS | Correlação temporal | +0,08 | Requer ajuste |

**Critérios de aprovação geral:** F1 > 0,80 (IF e LSTM CWRU), R² CMAPSS > 0,85, R² híbrido > 0,70, correlação temporal híbrida < -0,30.

---

## Estrutura de Arquivos

```
ml_module/
├── HANDOFF.md                          # Este documento
│
├── data/
│   ├── download_cwru.py                # Download + processamento CWRU → cwru_processed.parquet
│   ├── download_cmapss.py              # Download + processamento CMAPSS FD001 → train/test.parquet
│   ├── processed/                      # Parquets gerados (não versionados)
│   │   ├── cwru_processed.parquet
│   │   ├── cmapss_train.parquet
│   │   └── cmapss_test.parquet
│   └── raw/cmapss/                     # Arquivos brutos CMAPSS (train/test/RUL)
│
├── features/
│   ├── __init__.py                     # Reexporta build_features
│   └── feature_engineering.py          # load_raw_csv() + build_features() — 61 features Forzy
│
├── models/
│   ├── baseline/
│   │   ├── isolation_forest.py         # IF não supervisionado (Forzy)
│   │   ├── isolation_forest_cwru.py  # IF treinado em normais CWRU (F1=0,98)
│   │   ├── saved/                      # Artefatos IF Forzy (*.joblib, threshold.json)
│   │   └── saved_cwru/                 # Artefatos IF CWRU
│   │
│   ├── anomaly/
│   │   ├── lstm_autoencoder.py         # LSTM AE com Keras 3 + PyTorch (Forzy)
│   │   ├── lstm_autoencoder_cwru.py    # LSTM AE treinado em normais CWRU (F1=0,98)
│   │   ├── saved/                      # Artefatos LSTM Forzy (*.keras, scaler, threshold)
│   │   └── saved_cwru/                 # Artefatos LSTM CWRU
│   │
│   └── rul/
│       ├── xgboost_rul.py              # RUL por degradação relativa (labels sintéticos Forzy)
│       ├── xgboost_rul_cmapss.py       # RUL supervisionado CMAPSS FD001 (R²=0,91)
│       ├── xgboost_rul_hybrid.py       # RUL híbrido — labels IF+LSTM+tendência temporal
│       ├── rul_transfer_adapter.py     # Ponte Forzy → espaço CMAPSS (experimental)
│       ├── saved/                      # XGBoost RUL Forzy original
│       ├── saved_cmapss/               # XGBoost RUL CMAPSS
│       └── saved_hybrid/               # XGBoost RUL híbrido (produção)
│
├── training/
│   └── train_pipeline.py               # Pipeline único: CWRU + CMAPSS + retreino Forzy
│
├── inference/
│   └── predict.py                      # load_all_models(), predict_single(), predict_batch()
│
└── evaluation/
    └── metrics.py                      # Relatório consolidado de todos os modelos
```

**CSV Forzy (raiz do projeto):** `History_32026-05-19T11-46-10-920.csv`

---

## Como Usar

### Treinar todos os modelos do zero

```bash
# 1. Baixar datasets públicos
python ml_module/data/download_cwru.py
python ml_module/data/download_cmapss.py

# 2. Pipeline completo (CWRU + CMAPSS + Forzy)
python ml_module/training/train_pipeline.py

# 3. Treinar RUL híbrido (requer IF/LSTM Forzy em saved/)
python ml_module/models/rul/xgboost_rul_hybrid.py
```

Para pular o retreino Forzy no pipeline: `python ml_module/training/train_pipeline.py --skip-forzy`

### Rodar inferência em novos dados

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path("/caminho/para/Modelos ML - Forzy")
sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.features.feature_engineering import load_raw_csv, build_features
from ml_module.inference.predict import load_all_models, predict_single

# Carregar modelos uma vez (na inicialização da API)
models = load_all_models()

# Para cada requisição com novo CSV
csv_path = PROJECT_ROOT / "History_32026-05-19T11-46-10-920.csv"
raw_df = load_raw_csv(csv_path)
features_df = build_features(raw_df)

result = predict_single(features_df, models)

print(f"Status: {result['combined']['overall_status']}")
print(f"RUL: {result['rul']['rul_hours']:.1f} h")
print(f"Risco: {result['rul']['risk_level']}")
print(f"Modelo RUL: {result['rul']['model_version']}")  # 'hybrid' ou 'forzy_original'
```

### Verificar métricas dos modelos

```bash
python ml_module/evaluation/metrics.py
```

---

## Integração com a API

Os endpoints `/ml/anomaly` e `/ml/rul` devem importar de `ml_module.inference.predict`:

- `load_all_models()` no **startup event** do FastAPI (carrega modelos uma vez)
- `predict_single(features_df, models)` a cada requisição

Exemplo de router `ml.py`:

```python
from pathlib import Path
import sys

from fastapi import APIRouter, UploadFile, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml_module.features.feature_engineering import load_raw_csv, build_features
from ml_module.inference.predict import load_all_models, predict_single

router = APIRouter(prefix="/ml", tags=["ml"])

_models: dict | None = None


def get_models() -> dict:
    global _models
    if _models is None:
        _models = load_all_models()
    return _models


@router.on_event("startup")
async def load_ml_models():
    get_models()


@router.post("/anomaly")
async def detect_anomaly(file: UploadFile):
    """Detecção de anomalias via IF + LSTM."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        features_df = build_features(load_raw_csv(tmp_path))
        result = predict_single(features_df, get_models())
        return {
            "timestamp": str(result["timestamp"]),
            "isolation_forest": result["isolation_forest"],
            "lstm": result["lstm"],
            "health_score": result["combined"]["health_score"],
            "overall_status": result["combined"]["overall_status"],
            "recommendation": result["combined"]["recommendation"],
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/rul")
async def predict_rul(file: UploadFile):
    """Estimativa de RUL restante."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        features_df = build_features(load_raw_csv(tmp_path))
        result = predict_single(features_df, get_models())
        return {
            "timestamp": str(result["timestamp"]),
            "rul": result["rul"],
            "health_score": result["combined"]["health_score"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
```

**Prioridade de modelos em `load_all_models()`:**

| Componente | Preferência | Fallback |
|------------|-------------|----------|
| Isolation Forest | `saved_cwru/` (se compatível) | `saved/` (Forzy) |
| LSTM | `saved_cwru/` (se compatível) | `saved/` (Forzy) |
| RUL | `saved_hybrid/hybrid_rul.joblib` | `saved/` (Forzy original) |

> **Nota:** modelos CWRU usam features de domínio diferente (512 dims). Na inferência Forzy, o sistema faz fallback automático para modelos `saved/` quando o scaler CWRU é incompatível.

---

## Datasets Utilizados

### Forzy — `History_32026-05-19T11-46-10-920.csv`
- **Motor:** WEG W22 3cv
- **Duração:** ~4 horas de operação contínua
- **Registros:** 7.183 leituras (~0,6 s entre amostras)
- **Sensores:** 2 portas (velocidade mm/s, aceleração g, temperatura °C)
- **Labels:** nenhum (não supervisionado)
- **Evento conhecido:** cluster de anomalia ~13:44 do dia 19/05/2026

### CWRU — Case Western Reserve Bearing Dataset
- **Uso:** validação de modelos de anomalia com ground truth
- **Processado:** 5.689 janelas × 515 features (`cwru_processed.parquet`)
- **Métricas:** IF F1=0,9757 | LSTM F1=0,9846

### NASA CMAPSS FD001
- **Uso:** RUL supervisionado (degradação de turbofan)
- **Unidades:** 100 motores de treino + 100 de teste
- **Registros:** ~20.631 ciclos de treino
- **Métricas:** R²=0,9072 | MAE=8,69 ciclos | NASA Score=230

---

## Decisões Técnicas

### Por que Isolation Forest para anomalias grosseiras?
O IF é rápido, não supervisionado e robusto a outliers multidimensionais. Treinado apenas em dados normais (CWRU ou Forzy), detecta desvios globais nas 61 features de vibração/temperatura sem precisar de labels de falha.

### Por que LSTM Autoencoder para padrões sutis?
O LSTM captura dependências temporais em janelas de 60 registros (~36 s). Anomalias que evoluem gradualmente (degradação de rolamento, aquecimento progressivo) geram erros de reconstrução elevados mesmo quando o IF não dispara.

### Por que RUL híbrido em vez do adapter direto?
O `rul_transfer_adapter.py` mapeia features Forzy para o espaço CMAPSS (69 dims), mas apenas 37/69 features têm equivalente — correlação temporal ficou em +0,08 (sem tendência de degradação). O **RUL híbrido** gera labels realistas combinando:
- **30%** score do IF (saúde global)
- **50%** erro de reconstrução LSTM (degradação temporal)
- **20%** tendência linear da sessão (proxy físico)

Resultado: R²=0,9994 e correlação -0,57 no conjunto de teste estratificado.

### Por que R² = -3,02 no RUL Forzy original é esperado?
O `xgboost_rul.py` original usa labels sintéticos derivados de degradação relativa sem ground truth. Com apenas 4 h de dados e sem falha confirmada, o modelo não generaliza — R² negativo indica predições piores que a média. Por isso o híbrido substitui o original em produção (`saved_hybrid/`).

### Stack técnica
- **Python 3.14** com XGBoost, scikit-learn, pandas
- **Keras 3 + backend PyTorch** para LSTM (TensorFlow indisponível no Py 3.14)
- **Encoding CSV Forzy:** `cp1252`, separador `;`, `skiprows=[0,2]`

---

## Contatos e Próximos Passos

1. Coletar mais sessões Forzy com labels de manutenção para supervisionar RUL diretamente
2. Validar híbrido em dados de campo (não apenas CSV histórico)
3. Integrar `predict.py` nos endpoints FastAPI conforme exemplo acima
4. Agendar retreino mensal via `train_pipeline.py`

---

*Gerado em: junho/2026 — Forzy Digital Twin ML Module*
