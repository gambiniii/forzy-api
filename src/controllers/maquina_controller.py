from fastapi import Depends
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.enums import UserRoleEnum
from src.models.maquina import Maquina
from src.routers.auth import get_current_user, require_role
from src.routers.maquinas import MaquinaCreate, MaquinaUpdate
from src.services import maquina_service


def ctrl_list_maquinas(db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[Maquina]:
    return maquina_service.list_maquinas(db)


def ctrl_get_maquina(maquina_id: int, db: Session = Depends(get_db),
                     _=Depends(get_current_user)) -> Maquina:
    return maquina_service.get_maquina(db, maquina_id)


def ctrl_create_maquina(
    data: MaquinaCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Maquina:
    return maquina_service.create_maquina(db, **data.model_dump())


def ctrl_update_maquina(
    maquina_id: int,
    data: MaquinaUpdate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Maquina:
    return maquina_service.update_maquina(db, maquina_id, data.model_dump(exclude_none=True))


def ctrl_delete_maquina(
    maquina_id: int,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin)),
) -> None:
    maquina_service.delete_maquina(db, maquina_id)
