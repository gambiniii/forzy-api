from typing import Optional

from fastapi import Depends
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.models.atributo import Atributo, ComponenteAtributoValor
from src.models.enums import UserRoleEnum
from src.routers.atributos import AtributoCreate, ValorCreate, ValorUpdate
from src.routers.auth import get_current_user, require_role
from src.services import atributo_service


def ctrl_list_atributos(db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[Atributo]:
    return atributo_service.list_atributos(db)


def ctrl_create_atributo(
    data: AtributoCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> Atributo:
    return atributo_service.create_atributo(db, **data.model_dump())


def ctrl_delete_atributo(
    atributo_id: int,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin)),
) -> None:
    atributo_service.delete_atributo(db, atributo_id)


def ctrl_list_valores(
    componente_id: Optional[int] = None,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
) -> list[ComponenteAtributoValor]:
    return atributo_service.list_valores(db, componente_id)


def ctrl_create_valor(
    data: ValorCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> ComponenteAtributoValor:
    return atributo_service.create_valor(db, **data.model_dump())


def ctrl_update_valor(
    valor_id: int,
    data: ValorUpdate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> ComponenteAtributoValor:
    return atributo_service.update_valor(db, valor_id, data.model_dump(exclude_none=True))


def ctrl_delete_valor(
    valor_id: int,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
) -> None:
    atributo_service.delete_valor(db, valor_id)
