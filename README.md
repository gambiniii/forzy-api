# Forzy Digital Twin API
# RODAR NA VERSÃO 3.14.5 DO PYTHON!!!

## Como rodar localmente

### 1. Crie e ative a venv

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure o .env

```bash
cp .env.example .env
```

### 4. Suba os bancos

```bash
docker-compose up -d
```

### 5. Aplique as migrations e semeie dados de teste

Num banco novo (como o do docker-compose acima), a API sobe com o schema
certo mas **completamente vazia** — sem usuário pra logar, sem máquina
cadastrada, sem nenhuma leitura. Estes quatro scripts resolvem isso nessa
ordem, e são todos seguros de rodar mais de uma vez (idempotentes):

```bash
python scripts/aplicar_migrations.py     # cria as tabelas + aplica migrations/*.sql
python scripts/seed_forzy_motors.py      # cria Planta + Motor S1 (componente_id=2) + Motor S2 (componente_id=3)
python scripts/seed_usuario_teste.py     # cria login de teste: teste@forzy.com / forzy123
python scripts/seed_leituras_historico.py  # carrega ~4h de histórico REAL (não sintético) pros dois motores
```

O último script carrega o mesmo CSV do modo demonstração (abaixo), mas de uma
vez só e com os timestamps deslocados pra terminar "agora" — é o que dá pra
alguém abrir a tela e já ver gráfico, tendência de saúde e histórico de
diagnóstico populados, sem esperar nada acumular do zero. Sozinho ele não
mantém a leitura "viva" (o diagnóstico mais recente vai marcar sensor offline
depois de 30s sem leitura nova) — pra isso, combine com `SENSOR_MODE=demo`
abaixo.

### 6. Rode a API

```bash
uvicorn main:app --reload
```

Acesse: **http://localhost:8000/docs**

### Sem hardware disponível? Modo demonstração

Se o sensor físico da Forzy não estiver acessível (ou devolvendo dado
zerado), defina no `.env`:

```bash
SENSOR_MODE=demo
```

Isso troca o poller real por um replay do histórico real capturado
(`sensor/History_*.csv`) — ver `src/services/sensor_demo.py`. As leituras
ficam marcadas com `origem=demo` na tabela `leitura_sensor`, nunca se
confundem com dado real do hardware, e o frontend mostra um aviso de
"dados de demonstração" enquanto esse modo estiver ativo.

### Duas máquinas na apresentação: uma alimenta, outra só exibe

Cenário: um integrante roda a API com `SENSOR_MODE=demo` numa máquina
(alimenta o banco compartilhado), e você quer rodar sua própria instância só
pra exibir os dados, sem competir por escrever a mesma linha (poller/demo E
scheduler de diagnóstico rodando em dobro geram resultado inconsistente).
Aponte `POSTGRES_*` no seu `.env` pro mesmo banco e defina:

```bash
READ_ONLY=true
```

Essa instância não inicia nenhum processo em background que escreve no banco
— só serve a API pra leitura do que a outra instância for gravando. Testado
ao vivo: uma instância em `SENSOR_MODE=demo` escrevendo e outra em
`READ_ONLY=true` lendo o mesmo dado fresco, ambas no mesmo banco.

### Motores estáticos por condição (apresentação)

```bash
python scripts/seed_motores_demo_condicoes.py
```

Cria 7 máquinas na planta FIAP, uma por condição da tabela "Resultado
verificado" de `docs/resposta-pendencias-ml-fase2.md` (6 modos de falha
injetados + controle saudável), cada uma já com um diagnóstico congelado
(status, KPIs e — quando a condição tiver segmento associado — o destaque no
3D já vêm prontos). Não cria leitura nenhuma nem depende de nenhum processo
ao vivo — abre e já mostra o estado certo, sem esperar nada. Não toca nos
motores existentes (componente_id 1, 2, 3).
