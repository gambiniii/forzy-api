# Documentação Técnica — Forzy Digital Twin

**Repositórios:** `forzy-api-main` (backend) · `forzy-web` (frontend)

---

## 1. Arquitetura Geral

```
┌──────────────┐      REST       ┌──────────────────┐      SQL       ┌────────────┐
│  forzy-web   │ ───────────────▶│   forzy-api-main   │──────────────▶│  Postgres  │
│ (React/Vite) │◀─────────────── │   (FastAPI)        │◀───────────────│            │
└──────────────┘                 └──────────────────┘                └────────────┘
                                          │  ▲
                          HTTP polling    │  │ escreve
                     (a cada 10s)         ▼  │
                                  ┌──────────────────┐
                                  │  Endpoint Forzy    │
                                  │  (ngrok, externo)  │
                                  └──────────────────┘
```

- **Backend**: FastAPI + SQLAlchemy (ORM), Postgres como banco principal. Serve REST puro — não há WebSocket ativo no caminho de dados atual (existe um endpoint WS legado, ver seção 4.3).
- **Frontend**: React + TypeScript + Vite, styled-components, Chart.js. Busca dados via polling REST (sem WebSocket) a cada ~10s.
- **ML**: modelos treinados offline (Isolation Forest, LSTM Autoencoder, XGBoost), carregados em memória no startup do backend e executados sob demanda.
- **RAG/Chat**: agente conversacional (LangGraph) com acesso de leitura ao banco, tools próprias e geração de relatórios PDF/Excel/Word.

## 2. Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Backend | Python 3.14, FastAPI, SQLAlchemy, Pydantic |
| Banco de dados | PostgreSQL |
| Autenticação | JWT (python-jose) + bcrypt |
| ML | scikit-learn (Isolation Forest), Keras/TensorFlow (LSTM Autoencoder), XGBoost (RUL) |
| Frontend | React 18, TypeScript, Vite, styled-components, Chart.js, React Three Fiber (modelo 3D) |
| IA Conversacional | LangGraph, LangChain, OpenRouter (LLM) |

## 3. Modelo de Dados

Todas as tabelas usam nomenclatura em português como padrão do projeto.

```
planta ──< maquina ──< componente ──< leitura_sensor
                              │  ├──< diagnostico
                              │  ├──< especificacao_motor (1:1)
                              │  └──< componente_atributo_valor >── atributo
                              │
maquina ──< alerts
maquina ──< maintenance
```

| Tabela | Papel |
|---|---|
| `planta` | Instalação/cliente. |
| `maquina` | O motor. FK obrigatória para `planta`. |
| `componente` | Unidade monitorada dentro da máquina. FK obrigatória para `maquina` (`ON DELETE CASCADE`). |
| `especificacao_motor` | Placa de identificação (1:1 com `componente`): potência, tensão, corrente, rpm, frequência, polos, rendimento. |
| `atributo` / `componente_atributo_valor` | Modelo EAV para especificações técnicas adicionais e detalhadas (ex.: carcaça, grau de proteção, rolamentos) — usado quando a ficha padrão de `especificacao_motor` não é suficiente. |
| `leitura_sensor` | Série temporal de leituras (`componente_id`, `timestamp`, `temperatura`, `rpm`, `vibracao`). É a tabela principal lida pela API de sensores e pelo pipeline de ML. |
| `forzy_sensor_readings` | Cópia raw das leituras por porta física de sensor (auditoria/backup, não usada pela aplicação). |
| `diagnostico` | Um registro por execução do pipeline de ML: `overall_status`, `is_anomaly`, `lstm_severity`, `risk_level`, `rul_hours`, `maintenance_window_days`, `health_score`, `health_index`, `recommendation`. |
| `alerts` / `maintenance` | Alertas e registros de manutenção, vinculados à máquina (`motor_id → maquina.id`). |
| `users` | Usuários da aplicação, com `role` (`admin` / `technician` / `client`). |

Migrações são geridas por `migrate.py` na raiz do backend — um runner customizado (não Alembic) que executa uma lista ordenada de funções Python idempotentes contra o banco.

## 4. Pipeline de Dados

### 4.1 Ingestão (`src/services/forzy_poller.py`)

Um poller assíncrono, iniciado no `startup` da aplicação (`main.py`), consulta o endpoint físico da Forzy a cada 10 segundos:

- `GET /get_s1` → gravado em `leitura_sensor` sob um `componente_id` fixo (sensor físico 1).
- `GET /get_s2` → gravado sob outro `componente_id` fixo (sensor físico 2).

Cada porta de sensor corresponde a um motor físico **distinto** — não há médias entre os dois. Falhas de rede em uma leitura não impedem a gravação da outra, e o loop nunca é interrompido por uma exceção pontual (resiliência por ciclo).

### 4.2 Diagnóstico (`src/services/diagnostico_scheduler.py`)

Um segundo processo assíncrono, também iniciado no `startup`, roda a cada 5 minutos para **cada componente monitorado**:

1. Busca as últimas até 500 leituras do componente em `leitura_sensor`.
2. Se houver poucas leituras (< 5), pula o ciclo.
3. Converte as leituras em features (`ml_module/features/feature_engineering.py` — médias/desvios/máximos em janelas de 50/200/500 amostras) e roda a inferência combinada (`ml_module/inference/predict.py`).
4. Se o motor estiver desligado (seção 4 da doc de negócio), não salva diagnóstico.
5. Caso contrário, persiste o resultado em `diagnostico`.

Este pipeline é **desacoplado de qualquer conexão de frontend** — roda independentemente de haver alguém com o painel aberto.

### 4.3 Pipeline legado (WebSocket)

Existe um endpoint WebSocket (`/leituras/ws/{componente_id}`, em `src/routers/leituras.py`) desenhado originalmente para um dispositivo empurrar leituras individualmente. Ele mantém sua própria lógica de buffer e inferência ML in-memory, independente do scheduler da seção 4.2. Está fora do caminho principal de dados hoje, mas é reutilizável caso um dispositivo venha a fazer push direto no futuro.

## 5. API REST — Endpoints Principais

| Recurso | Rota base | Observações |
|---|---|---|
| Autenticação | `/auth` | `POST /login`, `POST /register`, `GET /me` (JWT Bearer). |
| Plantas | `/plantas` | CRUD. |
| Máquinas | `/maquinas` | CRUD. |
| Componentes | `/componentes` | CRUD, inclui `especificacao_motor` aninhada. |
| Atributos técnicos | `/atributos` | CRUD de atributos e valores (EAV). |
| Leituras | `/leituras` | Histórico via REST; endpoint WS legado (seção 4.3). |
| Sensores | `/sensors/component/{id}/latest` , `/sensors/component/{id}` | Leitura mais recente e histórico paginado — usados pelo frontend para os gráficos. |
| Diagnósticos | `/diagnosticos/componente/{id}` | Histórico de diagnósticos ML. |
| Análise agregada | `/analysis/{maquina_id}/report` | Relatório combinado (último diagnóstico + alertas + manutenções + última leitura) usado na tela de Gestão de Plantas. |
| Alertas / Manutenção | `/alerts`, `/maintenance` | CRUD, filtráveis por `motor_id`. |
| Chat / RAG | rotas do `rag_module` | Assistente conversacional, geração de relatórios. |

Toda rota (exceto login/registro) exige `Authorization: Bearer <token>`.

## 6. Frontend (`forzy-web`)

### 6.1 Roteamento e Inventário de Telas

Roteamento via `react-router-dom` (`src/App.tsx`), dentro de um `AppShell` (topbar + navegação lateral). A tabela abaixo indica, pra cada tela, se ela já consome dado real da API ou se ainda é uma tela ilustrativa (dado gerado localmente, sem chamada de backend) — informação relevante pra entender o estado atual de maturidade do produto:

| Rota | Tela | Propósito | Fonte de dado |
|---|---|---|---|
| `/` | Login | Autenticação | Real |
| `/dashboard` | Dashboard | Visão geral: KPIs, alertas e manutenções recentes | Parcial — KPIs/listas reais; os 2 gráficos de tendência usam dado ilustrativo |
| `/plants` | Gestão de Plantas | Lista de plantas com KPIs agregados por planta | Real |
| `/machinery` | Maquinário | Lista de máquinas de uma planta, com métricas ao vivo | Real |
| `/machinery/machine-detail/:id` | Detalhe do Motor | Ver seção 4 e a doc de negócio — tela central do produto | Real |
| `/machinery/equipment-form[/:id]` | Cadastro/Edição de Equipamento | Criar ou editar uma máquina | Real |
| `/machinery/alerts` | Alertas | Lista de alertas ativos/resolvidos | Real |
| `/machinery/maintenance` | Manutenção | Plano de manutenção preditiva | Real |
| `/machinery/assistant` | Assistente IA | Chat completo com o agente RAG | Real |
| `/machinery/realtime` | Tempo Real | Gauges/gráficos de uma máquina | Ilustrativa |
| `/machinery/history` | Histórico | Gráficos retroativos + log de eventos | Ilustrativa |
| `/machinery/health` | Saúde | Anéis de saúde consolidados por máquina | Ilustrativa |
| `/machinery/components` | Componentes | Grade de subcomponentes de uma máquina | Ilustrativa |
| `/machinery/component-analysis` | Análise de Componente | Espectro FFT, tendência de degradação | Ilustrativa |
| `/machinery/diagnosis` | Diagnóstico IA | Narrativa de causa-raiz, linha do tempo, plano de ação | Ilustrativa |
| `/machinery/reports` | Relatórios | Indicadores de falha/OEE | Ilustrativa |

As telas "ilustrativas" usam os helpers `randomData`/`makeLabels` (`src/components/charts/chartHelpers.ts`) para gerar série temporal fictícia — foram construídas para validar o design da interface e ficam como próximo passo natural de evolução do produto (ligar cada uma às mesmas fontes de dado já usadas na tela de Detalhe do Motor).

### 6.2 Sistema de Design (`src/components/ui/`)

Biblioteca de componentes própria (não usa Material UI/Ant/etc.), cada um em sua própria pasta com `Componente.tsx` + `Componente.styles.ts`:

`Card`, `Button`, `StatusPill`, `Input`/`Select`/`Textarea`, `Modal`, `Badge`, `MetricCard`, `AlertItem`, `MaintItem`, `HealthRing`, `ProbaRow`, `SpecRow`, `AiBubble`, `PageHeader`, `BackButton`, `Spinner`/`DotsLoader`/`Skeleton`, `Checkbox`/`Toggle`, `DatePicker`.

### 6.3 Shell de Layout (`src/components/layout/`)

- **AppShell** — casca raiz de toda rota; renderiza a `Topbar` e o conteúdo da página, além de um botão flutuante de assistente de IA arrastável e o popup `MiniChat`.
- **Topbar** — logo, relógio, alternância de tema claro/escuro, usuário logado, logout.
- **Sidebar** — navegação principal (Gestão de Plantas / Maquinário).
- **MiniChat** — janela de chat flutuante, ciente da tela atual, que pode ser expandida para a tela completa do Assistente.

### 6.4 Gerenciamento de Estado

Não há Redux nem uma camada de cache de servidor (React Query/SWR): cada domínio tem seu próprio hook (`useMaquinas`, `useAlertas`, `useManutencao`, `usePlantas`, `usePlantStats`, `useLiveLeituras`, etc., em `src/hooks/`) que busca dado via `fetch` e guarda em `useState` local.

Três contextos React cobrem apenas estado de UI (nunca dado de servidor):
- **NavigationContext** — camada fina sobre `useNavigate` do react-router, expondo `goTo(tela, id?)`/`goBack()` com nomes de tela tipados, em vez de strings de rota soltas pelo código.
- **ThemeContext** — alterna o atributo `data-theme` no `<html>`, consumido pelas variáveis CSS (seção 6.5).
- **SidebarContext** — estado de colapso da barra lateral.

### 6.5 Estilização

Toda a paleta e tokens (cores, raio de borda, largura da sidebar) são variáveis CSS globais em `src/styles/globals.css`, com um bloco `[data-theme="light"]` sobrescrevendo os mesmos nomes para o tema claro. Os componentes `styled-components` consomem essas variáveis diretamente (`var(--bg2)`, `var(--text1)` etc.) — não há um objeto de tema JS do styled-components. Os gráficos Chart.js têm sua própria paleta centralizada em `chartHelpers.ts`, já que não conseguem ler variáveis CSS diretamente.

### 6.6 Autenticação (fluxo completo)

1. `LoginScreen` chama `auth.service.login(email, senha)` → `POST /auth/login`.
2. O token retornado é salvo em `localStorage` (`services/api.ts`).
3. Toda chamada subsequente do wrapper `fetch` (`services/api.ts`) anexa `Authorization: Bearer <token>` automaticamente.
4. Um `401` do backend limpa o token e força um redirecionamento completo para `/`.
5. `useCurrentUser` busca `GET /auth/me` pra popular o avatar/usuário exibido na Topbar/Sidebar.

No backend, o token é um JWT assinado (`HS256`), com validade em `ACCESS_TOKEN_EXPIRE_MINUTES` (`.env`), carregando `sub` (e-mail) e `role`. Rotas sensíveis usam `require_role(...)` para restringir a `admin`/`technician`.

### 6.7 Build e Tooling

Vite 5 + React 19 + TypeScript, sem alias de caminho configurado (todos os imports são relativos). A URL da API é resolvida em tempo de execução via `VITE_API_URL` (padrão `http://localhost:8000`), sem proxy do Vite — CORS é tratado no backend.

## 7. Assistente de IA (RAG)

Localizado em `rag_module/`. Um agente LangGraph com acesso a tools que consultam o banco diretamente (`rag_module/tools/db_tool.py`) e geram relatórios em PDF/Excel/Word (`rag_module/tools/report_tool.py`). O carregamento do módulo é opcional — se as dependências de LLM não estiverem instaladas, a aplicação sobe normalmente sem o chat (ver `main.py`, import protegido por `try/except`).

## 8. Configuração e Execução

Variáveis de ambiente relevantes (`.env`, ver `src/config.py`):

| Variável | Uso |
|---|---|
| `POSTGRES_*` | Conexão com o banco. |
| `SECRET_KEY` / `ALGORITHM` / `ACCESS_TOKEN_EXPIRE_MINUTES` | Configuração do JWT. |
| `FORZY_SENSOR_URL` | Endpoint externo (ngrok) consultado pelo poller. |
| `OPENROUTER_API_KEY` / `LLM_MODEL` | Modelo usado pelo assistente de IA. |

Subir localmente:
```bash
pip install -r requirements.txt
uvicorn main:app --reload      # backend, porta 8000
npm run dev                     # frontend (forzy-web), porta 5173
```
