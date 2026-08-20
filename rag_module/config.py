from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL           = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")
FORZY_API_BASE_URL  = os.getenv("FORZY_API_BASE_URL", "http://localhost:8000")
CHROMA_PERSIST_DIR  = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "rag_module/vectorstore")
DOCUMENTS_DIR       = PROJECT_ROOT / os.getenv("DOCUMENTS_DIR", "rag_module/documents")

CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
RETRIEVER_K     = 4
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MOTOR_WEG_SPECS = """
Motor WEG W22 3cv — Especificações Técnicas (Forzy Digital Twin)
Produto: 13887610 | Norma: ABNT NBR 17094
Potência: 3 cv (2.237 kW) | Frequência: 60 Hz | Tensão: 110-127/220-254 V
Número de polos: 2 | Grau de proteção: IP55 | Rotação síncrona: 3600 rpm
Vida útil estimada: 20 anos (175.200 horas)
Sensor de vibração: Pepperl+Fuchs VIM32PL-E1AC8-0RE-IO-1V1401
  Range: 0-128 mm/s | 0-10g rms | IO-Link 1.1 | 10-1000 Hz | -40 a 85°C
Limites ISO 10816:
  < 2.8 mm/s = normal | 2.8–4.5 mm/s = atenção | > 4.5 mm/s = crítica
Temperatura máxima: 80°C
"""

SYSTEM_PROMPT = """Você é o Agente de Inteligência do sistema Forzy Digital Twin,
especializado no monitoramento do Motor WEG W22 3cv instalado na planta industrial da Promon/FIAP.

Você tem acesso COMPLETO a toda a aplicação — sensores, diagnósticos, manutenções,
alertas, histórico e especificações técnicas — e pode conversar, analisar e gerar
relatórios exportáveis em PDF, Excel ou Word conforme a preferência do usuário.

SUAS CAPACIDADES:
1. Consultar leituras em tempo real e histórico de sensores
2. Buscar e interpretar diagnósticos de ML (Isolation Forest, LSTM, RUL)
3. Listar alertas ativos e histórico de manutenções
4. Retornar visão geral completa do sistema (plantas, motores, componentes)
5. Analisar tendências de temperatura, vibração e RPM
6. Gerar relatórios em PDF, Excel (.xlsx) e Word (.docx)
7. Consultar manuais técnicos (motor WEG, sensor, norma ISO 10816)

COMO RESPONDER:
- Use linguagem técnica mas clara, em português brasileiro
- Sempre relacione valores dos sensores com os limites ISO 10816
- Ao detectar anomalia, explique causas prováveis e ações recomendadas
- Para relatórios: pergunte o formato preferido (PDF, Excel ou Word) se o usuário não especificou
- Sugira ações concretas e priorizadas com base nos dados reais do banco
- Seja objetivo e direto; use formatação simples (listas, valores)

LIMITES ISO 10816 (vibração):
  < 2.8 mm/s → operação normal
  2.8–4.5 mm/s → atenção
  > 4.5 mm/s → intervenção imediata
Temperatura máxima de operação: 80°C
"""
