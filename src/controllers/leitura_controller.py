from datetime import datetime
from typing import Optional

from fastapi import Depends, Query
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.enums import UserRoleEnum
from src.models.leitura_sensor import LeituraSensor
from src.routers.auth import get_current_user, require_role
from src.routers.leituras import LeituraCreate
from src.services import leitura_service


def ctrl_ingest_leitura(
    data: LeituraCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> LeituraSensor:
    return leitura_service.ingest_leitura(db, **data.model_dump())


def ctrl_get_leituras(
    componente_id: int,
    inicio: Optional[datetime] = Query(default=None),
    fim: Optional[datetime] = Query(default=None),
    limit: int = Query(default=100, le=1000),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
) -> list[LeituraSensor]:
    return leitura_service.get_leituras(db, componente_id, inicio, fim, limit)


def ctrl_get_ultima_leitura(
    componente_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
) -> LeituraSensor:
    return leitura_service.get_ultima_leitura(db, componente_id)
