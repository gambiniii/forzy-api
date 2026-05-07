# Forzy Digital Twin API

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

### 5. Rode a API

```bash
uvicorn main:app --reload
```

Acesse: **http://localhost:8000/docs**
