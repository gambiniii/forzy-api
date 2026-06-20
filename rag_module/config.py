from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
FORZY_API_BASE_URL = os.getenv("FORZY_API_BASE_URL", "http://localhost:8000")
CHROMA_PERSIST_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "rag_module/vectorstore")
DOCUMENTS_DIR = PROJECT_ROOT / os.getenv("DOCUMENTS_DIR", "rag_module/documents")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVER_K = 4
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MOTOR_WEG_SPECS = """
Motor WEG W22 3cv — Especificações Técnicas (Forzy Digital Twin)
Produto: 13887610
Norma: ABNT NBR 17094
Potência: 3 cv (2.237 kW)
Frequência: 60 Hz
Tensão: 110-127/220-254 V
Número de polos: 2
Grau de proteção: IP55
Rotação síncrona: 3600 rpm
Fixação: Com pés
Forma construtiva: B3D
Refrigeração: IC411 - TFVE
Limites ISO 10816:
  Velocidade vibração < 2.8 mm/s = operação boa
  Velocidade vibração < 4.5 mm/s = operação aceitável
  Velocidade vibração > 4.5 mm/s = requer atenção imediata
Temperatura máxima de operação: 80°C
Vida útil estimada: 20 anos (175.200 horas)
Sensor de vibração: Pepperl+Fuchs VIM32PL-E1AC8-0RE-IO-1V1401
  Range vibração: 0-128 mm/s | 0-10g rms
  Interface: IO-Link 1.1 | Frequência: 10-1000 Hz
  Temperatura: -40 a 85°C
"""

SYSTEM_PROMPT = """Você é o assistente técnico do sistema Forzy Digital Twin, 
especializado no monitoramento do motor WEG W22 3cv instalado na planta industrial.

Seu papel é ajudar técnicos de manutenção a:
- Interpretar dados dos sensores de vibração e temperatura
- Diagnosticar anomalias detectadas pelos modelos de ML
- Consultar manuais técnicos do motor e dos sensores
- Recomendar ações de manutenção preventiva
- Estimar vida útil restante (RUL) e janelas de manutenção

Ao responder:
- Use linguagem técnica mas clara, em português brasileiro
- Sempre relacione os dados dos sensores com os limites ISO 10816
- Se houver anomalia detectada, explique o que pode estar causando
- Sugira ações concretas e priorizadas
- Quando citar valores dos sensores, informe se estão dentro ou fora do normal
- Baseie respostas nos documentos técnicos e nos dados em tempo real disponíveis

Limites de referência do motor:
- Vibração normal: < 2.8 mm/s
- Vibração aceitável: 2.8 a 4.5 mm/s  
- Vibração crítica: > 4.5 mm/s
- Temperatura normal de operação: até 80°C
"""
