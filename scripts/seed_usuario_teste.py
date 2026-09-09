"""
Cria um usuário de teste pra login — um banco novo não tem nenhum usuário
cadastrado, e o frontend exige login pra mostrar qualquer coisa.

Uso: python scripts/seed_usuario_teste.py
Idempotente — não duplica se o e-mail já existir.

Credenciais fixas de teste, pensadas pra ambiente isolado de avaliação —
troque a senha se este script rodar contra um banco que não seja descartável.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.postgres import SessionLocal, create_tables
from src.models.enums import UserRoleEnum
from src.models.user import User
from src.routers.auth import hash_password

EMAIL = "teste@forzy.com"
SENHA = "forzy123"
NOME = "Usuário de Teste"


def run() -> None:
    create_tables()
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == EMAIL).first():
            print(f"Usuário {EMAIL} já existe — pulando.")
            return
        db.add(User(
            name=NOME, email=EMAIL,
            hashed_password=hash_password(SENHA),
            role=UserRoleEnum.admin,
        ))
        db.commit()
        print(f"Usuário criado: {EMAIL} / senha: {SENHA} (role=admin)")
    finally:
        db.close()


if __name__ == "__main__":
    run()
