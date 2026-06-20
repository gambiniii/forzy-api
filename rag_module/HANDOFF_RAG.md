# Forzy Digital Twin — RAG Module Handoff

## Visão Geral
Sistema conversacional com RAG (Retrieval-Augmented Generation) e
agente LangGraph para troubleshooting técnico do motor WEG W22 3cv.
LLM: google/gemini-2.5-flash via OpenRouter.

## Arquitetura

```
Usuario → POST /chat/message
           ↓
      LangGraph Agent
      ├── get_sensor_status()     → API Forzy /sensors/latest
      ├── get_ml_analysis()       → Modelos ML locais (IF+LSTM+RUL)
      ├── get_maintenance_history() → API Forzy /maintenance
      ├── get_active_alerts()     → API Forzy /alerts
      └── search_technical_docs() → ChromaDB (PDFs + TXTs)
           ↓
      Gemini 2.5 Flash (OpenRouter)
           ↓
      Resposta em linguagem natural
```

## Estrutura de Arquivos

```
rag_module/
├── config.py              Variáveis de ambiente e constantes
├── documents/             Corpus do RAG (PDFs e TXTs)
│   ├── sensor_vibracao.pdf      Datasheet Pepperl+Fuchs VIM32PL
│   ├── desafio_forzy.pdf        Especificações do projeto
│   ├── motor_weg_w22.txt        Specs WEG W22 3cv + limites ISO
│   ├── iso_10816_referencia.txt Norma de avaliação de vibração
│   ├── faq_manutencao.txt       FAQ de manutenção preventiva
│   └── severidade_ml.txt        Guia de severidade IF/LSTM/XGBoost
├── vectorstore/           ChromaDB persistido (embeddings)
├── memory/                Histórico por session_id (JSON)
├── logs/
│   └── agent.log          Log estruturado JSON
├── chains/
│   ├── vectorstore.py     Indexação e retrieval de documentos
│   ├── rag_chain.py       RAG chain com resiliência
│   ├── agent.py           LangGraph agent + memória + logs
│   ├── memory.py          Persistência de sessões
│   ├── logger.py          JsonFormatter + CallTimer
│   └── llm_resilience.py  Retry + timeout + fallback
├── tools/
│   └── sensor_tool.py     4 tools: sensores, ML, manutenção, alertas
└── api/
    ├── router.py          FastAPI router — 7 endpoints /chat/*
    ├── test_router.py     Teste standalone sem subir FastAPI
    └── test_health.py     Teste do endpoint /chat/health
```

## Como Usar

### Variáveis de ambiente (.env na raiz do projeto):
```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.5-flash
FORZY_API_BASE_URL=http://localhost:8000
CHROMA_PERSIST_DIR=rag_module/vectorstore
DOCUMENTS_DIR=rag_module/documents
```

### Registrar no FastAPI (src/main.py do forzy-api):
```python
import asyncio
from rag_module.api.router import router as chat_router, warmup

app.include_router(chat_router)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(warmup_rag())

async def warmup_rag():
    await warmup()
```

### Exemplo de request (frontend → backend):
```json
{
  "message": "Como está o motor agora? Tem alguma anomalia?",
  "machine_id": "1",
  "session_id": "tecnico_01",
  "history": [],
  "mode": "agent"
}
```

### Exemplo de response:
```json
{
  "answer": "Motor operando com vibração normal (2.1 mm/s)...",
  "tools_used": ["get_sensor_status", "get_ml_analysis"],
  "sources": [],
  "mode": "agent",
  "machine_id": "1",
  "session_id": "tecnico_01",
  "fallback_used": false,
  "response_time_ms": 7432
}
```

### Manter histórico de conversa:
O histórico é persistido automaticamente em disco por `session_id`.
Não é necessário enviar `history` a cada turno — basta reutilizar o mesmo `session_id`:

```python
session_id = "tecnico_01"

# Turno 1
requests.post("/chat/message", json={
    "message": "Como está o motor?",
    "session_id": session_id,
    "mode": "agent"
})

# Turno 2 — histórico carregado automaticamente do disco
requests.post("/chat/message", json={
    "message": "E o RUL?",
    "session_id": session_id,
    "mode": "agent"
})
```

Alternativa manual (sem persistência):
```python
history = []
result = requests.post("/chat/message", json={
    "message": "Como está o motor?",
    "history": history,
    "mode": "agent"
})
history = result.json().get("messages", [])
```

## Modos de Operação

| Modo | Quando usar | Tools ativas |
|------|-------------|--------------|
| agent | Chat geral com dados em tempo real | Todas as 5 tools |
| rag | Consulta apenas documentos técnicos | Apenas vectorstore |

## Fallback quando API offline
Todas as tools têm fallback automático:
- get_sensor_status → dados simulados realistas
- get_ml_analysis → roda modelos ML locais diretamente
- get_maintenance_history → histórico simulado
- get_active_alerts → retorna mensagem de API offline

## Adicionar novos documentos ao RAG
1. Copiar PDF ou TXT para rag_module/documents/
2. Reindexar: python rag_module/chains/vectorstore.py --rebuild
3. Reiniciar a API

## Custo estimado (OpenRouter - google/gemini-2.5-flash)
- Input: ~$0.15 por milhão de tokens
- Output: ~$0.60 por milhão de tokens
- Estimativa por conversa de 5 turnos: ~$0.002
- Com $5.00 disponíveis: ~2.500 conversas de demo

## Integração com o Frontend
O frontend deve:
1. Gerar e reutilizar `session_id` por sessão de chat (persistência automática)
2. Limpar histórico via DELETE /chat/sessions/{id} ao iniciar nova sessão
3. Exibir tools_used como badge (ex: "Consultou sensores + ML")
4. Exibir `fallback_used: true` como aviso quando LLM estiver indisponível
5. Modo "rag" para botão "Consultar manual"
6. Modo "agent" para chat geral

## Status do Módulo

| Item | Detalhe | Status |
|------|---------|--------|
| Vectorstore | 37 chunks, 6 documentos | OK |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | OK |
| LLM | google/gemini-2.5-flash via OpenRouter | OK |
| Agent | LangGraph com 5 tools | OK |
| Fallback | RAG local sem LLM | OK |
| Memória | Persistência por session_id em disco | OK |
| Resiliência | 3 retries + timeout + fallback | OK |
| Logs | JSON estruturado em agent.log | OK |
| IF CWRU F1 | 0.9757 | APROVADO |
| LSTM CWRU F1 | 0.9846 | APROVADO |
| XGBoost R² | 0.9994 (híbrido) | APROVADO |

## Melhorias Implementadas

### 1. Embedding Multilíngue
Modelo trocado de `all-MiniLM-L6-v2` para
`paraphrase-multilingual-MiniLM-L12-v2`.
Resultado: queries em português agora recuperam chunks do
`sensor_vibracao.pdf` (inglês) corretamente.

### 2. Corpus Rico (6 documentos, 37 chunks)
Adicionados 2 documentos ao RAG:
- `faq_manutencao.txt` — 8 perguntas e respostas de manutenção
  preventiva, limites de vibração/temperatura e protocolo de alertas
- `severidade_ml.txt` — guia completo dos níveis de severidade
  do IF, LSTM e XGBoost com ações recomendadas por nível

### 3. Persistência de Histórico entre Sessões
Histórico de conversa salvo em `rag_module/memory/{session_id}.json`.
Máximo de 10 turnos por sessão. Carregado automaticamente ao reabrir o chat.
Novos endpoints: GET/DELETE /chat/sessions/{session_id}

### 4. Retry Automático e Fallback
3 tentativas com backoff exponencial (2s, 4s, 8s).
Timeout de 15s para RAG chain e 20s para o agent.
Se todas as tentativas falharem: resposta baseada nos documentos locais
sem depender do LLM. Campo `fallback_used` na response indica qual caminho foi tomado.

### 5. Endpoint /chat/health Enriquecido
Retorna em JSON:
- Status do vectorstore (chunks, modelo de embedding)
- Status dos modelos ML (IF F1, LSTM F1, XGBoost R²)
- Sessões salvas (total + recentes)
- Documentos indexados (nome, tamanho, tipo)
- Configuração OpenRouter (modelo, timeout, retries)

### 6. Logs Estruturados em JSON
Arquivo: `rag_module/logs/agent.log`
Cada chamada registra: timestamp, session_id, question_preview,
tools_used, response_time_ms, tokens_estimated, fallback_used.
Novo endpoint GET /chat/logs?n=20 para monitoramento em tempo real.

## Endpoints Disponíveis

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | /chat/health | Status completo do sistema |
| POST | /chat/message | Enviar mensagem (agent ou rag) |
| POST | /chat/warmup | Pré-carregar modelos |
| GET | /chat/sessions | Listar sessões salvas |
| GET | /chat/sessions/{id} | Resumo de uma sessão |
| DELETE | /chat/sessions/{id} | Limpar histórico |
| GET | /chat/logs?n=20 | Últimas entradas do log |

## Estrutura Final do rag_module/

```
rag_module/
├── config.py
├── HANDOFF_RAG.md
├── documents/          6 arquivos (2 PDFs + 4 TXTs)
├── vectorstore/        ChromaDB — 37 chunks persistidos
├── memory/             Histórico por session_id (JSON)
├── logs/
│   └── agent.log       Log estruturado JSON
├── chains/
│   ├── vectorstore.py  Indexação e retrieval
│   ├── rag_chain.py    RAG chain com resiliência
│   ├── agent.py        LangGraph agent + memória + logs
│   ├── memory.py       Persistência de sessões
│   ├── logger.py       JsonFormatter + CallTimer
│   └── llm_resilience.py  Retry + timeout + fallback
├── tools/
│   └── sensor_tool.py  4 tools com fallback automático
└── api/
    ├── router.py       7 endpoints FastAPI
    ├── test_router.py
    ├── test_health.py
    └── test_resilience.py (em chains/)
```

*Gerado em: junho/2026 — Forzy Digital Twin RAG Module (v2.0)*
