from typing import Optional

from fastapi import Depends
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.componente import Componente, EspecificacaoMotor
from src.models.enums import UserRoleEnum
from src.routers.auth import get_current_user, require_role
from src.routers.componentes import ComponenteCreate, ComponenteUpdate, EspecificacaoMotorSchema
from src.services import componente_service


def ctrl_list_componentes(
    maquina_id: Optional[int] = None,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
) -> list[Componente]:
    return componente_service.list_componentes(db, maquina_id)


def ctrl_get_componente(componente_id: int, db: Session = Depends(get_db),
                        _=Depends(get_current_user)) -> Componente:
    return componente_service.get_componente(db, componente_id)


def ctrl_create_componente(
    data: ComponenteCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Componente:
    esp = data.especificacao_motor.model_dump(exclude_none=True) if data.especificacao_motor else None
    return componente_service.create_componente(
        db,
        maquina_id=data.maquina_id,
        nome=data.nome,
        tipo=data.tipo,
        fabricante=data.fabricante,
        data_instalacao=data.data_instalacao,
        status=data.status,
        especificacao_motor=esp,
    )


def ctrl_update_componente(
    componente_id: int,
    data: ComponenteUpdate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Componente:
    return componente_service.update_componente(db, componente_id, data.model_dump(exclude_none=True))


def ctrl_upsert_especificacao_motor(
    componente_id: int,
    data: EspecificacaoMotorSchema,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> EspecificacaoMotor:
    return componente_service.upsert_especificacao_motor(db, componente_id, data.model_dump(exclude_none=True))


def ctrl_delete_componente(
    componente_id: int,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin)),
) -> None:
    componente_service.delete_componente(db, componente_id)
