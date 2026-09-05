from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from src.config import settings

engine = create_engine(settings.postgres_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency injetada nas rotas para obter sessão do banco."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Cria todas as tabelas ao iniciar a aplicação."""
    from src.models.user import User  # noqa
    from src.models.planta import Planta  # noqa
    from src.models.maquina import Maquina  # noqa
    from src.models.componente import Componente, EspecificacaoMotor, ComponenteLimite  # noqa
    from src.models.atributo import Atributo, ComponenteAtributoValor  # noqa
    from src.models.leitura_sensor import LeituraSensor  # noqa
    from src.models.alert import Alert  # noqa
    from src.models.maintenance import Maintenance  # noqa
    from src.models.audit_log import AuditLog  # noqa

    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[DB] Falha ao conectar no banco: {e}")
        print("[DB] API subindo sem banco — endpoints que dependem do DB vão retornar erro 503.")
