"""
Aplica, em ordem, as migrations de migrations/*.sql relevantes para um banco
novo (ex: docker-compose up + create_tables + este script + seed_forzy_motors.py).

Todas as migrations aqui são IF NOT EXISTS / ON CONFLICT — seguro rodar de
novo contra um banco que já tem algumas ou todas aplicadas.

DELIBERADAMENTE PULA 002_motor_weg_w22_dados_completos.sql: aquela migration
alimenta dados de um componente_id=1 legado (o motor de demonstração da FIAP,
sem relação com o par S1/S2 do Forzy) e nem é idempotente (o INSERT em
componente_atributo_valor não tem ON CONFLICT) — rodar num banco novo, sem
esse componente_id=1 já existindo, quebra com violação de foreign key. Só
mexe nisso quem estiver reproduzindo o histórico completo do projeto, não
quem só precisa do par S1/S2 funcionando.

Uso: python scripts/aplicar_migrations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.db.postgres import SessionLocal, create_tables

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

PULAR = {"002_motor_weg_w22_dados_completos.sql"}


def run() -> None:
    create_tables()  # tabelas base (diagnostico, componente_limite, leitura_sensor...)
                      # precisam existir antes de qualquer ALTER TABLE abaixo.
    arquivos = sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.name not in PULAR)
    db = SessionLocal()
    try:
        for caminho in arquivos:
            print(f"Aplicando {caminho.name}...")
            db.execute(text(caminho.read_text(encoding="utf-8")))
            db.commit()
            print("  OK.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    print(f"{len(arquivos)} migration(s) aplicada(s).")


if __name__ == "__main__":
    run()
