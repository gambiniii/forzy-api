"""
Seed reproduzível dos motores Forzy S1/S2 — Planta + Máquina + Componente +
Especificação, nos IDs que o resto do código espera fixos:

    componente_id=2 -> S1   componente_id=3 -> S2

(ver COMPONENTE_ID_S1/S2 em src/services/forzy_poller.py e
src/services/sensor_demo.py, e COMPONENTE_IDS em
src/services/diagnostico_scheduler.py).

Sem isso, um banco novo (docker-compose up + create_tables) sobe com o schema
vazio — a API sobe, mas não tem nenhuma máquina cadastrada, e o poller (ou o
modo demonstração) falha com violação de foreign key ao tentar gravar uma
leitura para um componente que não existe.

Não toca no componente_id=1 (motor genérico legado da FIAP, alimentado pela
migration 002/migrate.py) — isso é uma história separada, sem relação com o
par S1/S2 do Forzy.

Uso: python scripts/seed_forzy_motors.py
Idempotente — roda de novo sem duplicar se os componentes já existirem.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.db.postgres import SessionLocal, create_tables
from src.models.componente import Componente, EspecificacaoMotor
from src.models.enums import StatusEnum
from src.models.maquina import Maquina
from src.models.planta import Planta

COMPONENTE_ID_S1 = 2
COMPONENTE_ID_S2 = 3

# Motor WEG W22 3cv monofásico — mesma placa técnica usada em migrate.py
# (migration_003_motor_weg_w22_monofasico), reaproveitada aqui pros dois motores.
ESPECIFICACAO_W22 = dict(
    potencia_kw=2.2, tensao_nominal=220, corrente_nominal=12,
    rpm_nominal=3525, frequencia_hz=60, numero_polos=2, rendimento=82,
)


def _inserir_componente_com_id_fixo(db, *, id_desejado: int, maquina_id: int, nome: str) -> None:
    """INSERT direto (fora do ORM) pra controlar o ID — o SERIAL da tabela não
    escolhe sozinho. Realinha a sequence depois pra não colidir com o próximo
    insert automático (ex: uma máquina cadastrada pela tela depois do seed)."""
    db.execute(text("""
        INSERT INTO componente (id, maquina_id, nome, tipo, fabricante, data_instalacao, status)
        VALUES (:id, :maquina_id, :nome, 'Motor', 'WEG', :data_instalacao, :status)
    """), {
        "id": id_desejado, "maquina_id": maquina_id, "nome": nome,
        "data_instalacao": date(2024, 1, 10), "status": StatusEnum.active.value,
    })
    db.execute(text(
        "SELECT setval('componente_id_seq', (SELECT MAX(id) FROM componente))"
    ))


def _planta_padrao(db) -> Planta:
    planta = db.query(Planta).order_by(Planta.id).first()
    if planta:
        return planta
    planta = Planta(nome="Planta Principal", ativo=True)
    db.add(planta)
    db.flush()
    print(f"  Planta criada: {planta.nome} (id={planta.id})")
    return planta


def seed_motor(db, *, nome_maquina: str, nome_componente: str, componente_id: int) -> None:
    if db.query(Componente).filter(Componente.id == componente_id).first():
        print(f"  componente_id={componente_id} já existe — pulando.")
        return

    planta = _planta_padrao(db)

    maquina = db.query(Maquina).filter(Maquina.nome == nome_maquina).first()
    if not maquina:
        maquina = Maquina(
            nome=nome_maquina, tipo="Motor Elétrico Monofásico", fabricante="WEG",
            ano_instalacao=2024, status=StatusEnum.active, planta_id=planta.id,
        )
        db.add(maquina)
        db.flush()

    _inserir_componente_com_id_fixo(db, id_desejado=componente_id, maquina_id=maquina.id, nome=nome_componente)
    db.add(EspecificacaoMotor(componente_id=componente_id, **ESPECIFICACAO_W22))
    db.commit()
    print(f"  Criado: {nome_maquina} / {nome_componente} (componente_id={componente_id})")


def run() -> None:
    create_tables()
    db = SessionLocal()
    try:
        print("Semeando Motor WEG W22 — Unidade S1...")
        seed_motor(db, nome_maquina="Motor WEG W22 - Unidade S1",
                   nome_componente="Motor WEG W22 3cv Monofásico - S1", componente_id=COMPONENTE_ID_S1)
        print("Semeando Motor WEG W22 — Unidade S2...")
        seed_motor(db, nome_maquina="Motor WEG W22 - Unidade S2",
                   nome_componente="Motor WEG W22 3cv Monofásico - S2", componente_id=COMPONENTE_ID_S2)
    finally:
        db.close()
    print("Seed concluído.")


if __name__ == "__main__":
    run()
