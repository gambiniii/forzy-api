from fastapi import Depends
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.enums import UserRoleEnum
from src.models.planta import Planta
from src.routers.auth import get_current_user, require_role
from src.routers.plantas import PlantaCreate, PlantaUpdate
from src.services import planta_service


def ctrl_list_plantas(db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[Planta]:
    return planta_service.list_plantas(db)


def ctrl_get_planta(planta_id: int, db: Session = Depends(get_db),
                    _=Depends(get_current_user)) -> Planta:
    return planta_service.get_planta(db, planta_id)


def ctrl_create_planta(
    data: PlantaCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Planta:
    return planta_service.create_planta(db, **data.model_dump())


def ctrl_update_planta(
    planta_id: int,
    data: PlantaUpdate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Planta:
    return planta_service.update_planta(db, planta_id, data.model_dump(exclude_none=True))


def ctrl_delete_planta(
    planta_id: int,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin)),
) -> None:
    planta_service.delete_planta(db, planta_id)
