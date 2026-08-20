from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.postgres import get_db
from src.routers.auth import get_current_user, require_role
from src.models.enums import UserRoleEnum

router = APIRouter()


class AtivoOut(BaseModel):
    id: int
    name: str
    location: Optional[str] = None
    description: Optional[str] = None
    status: str
    created_at: str


class AtivoCreate(BaseModel):
    name: str
    location: Optional[str] = None
    cidade: Optional[str] = None
    estado: Optional[str] = None
    description: Optional[str] = None
    status: str = "active"


@router.get("/", response_model=list[AtivoOut])
def list_ativos(db: Session = Depends(get_db), _=Depends(get_current_user)):
    rows = db.execute(text(
        "SELECT id, name, location, description, status, created_at FROM ativos ORDER BY id"
    )).fetchall()
    return [
        AtivoOut(
            id=r[0], name=r[1], location=r[2],
            description=r[3], status=r[4],
            created_at=r[5].isoformat() if r[5] else "",
        )
        for r in rows
    ]


@router.get("/{ativo_id}", response_model=AtivoOut)
def get_ativo(ativo_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    row = db.execute(
        text("SELECT id, name, location, description, status, created_at FROM ativos WHERE id = :id"),
        {"id": ativo_id}
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Ativo não encontrado")
    return AtivoOut(
        id=row[0], name=row[1], location=row[2],
        description=row[3], status=row[4],
        created_at=row[5].isoformat() if row[5] else "",
    )


@router.post("/", response_model=AtivoOut, status_code=201)
def create_ativo(
    data: AtivoCreate,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRoleEnum.admin, UserRoleEnum.technician)),
):
    row = db.execute(
        text("""
            INSERT INTO ativos (name, location, cidade, estado, description, status, ativo)
            VALUES (:name, :location, :cidade, :estado, :description, :status, true)
            RETURNING id, name, location, description, status, created_at
        """),
        data.model_dump()
    ).fetchone()
    db.commit()
    return AtivoOut(
        id=row[0], name=row[1], location=row[2],
        description=row[3], status=row[4],
        created_at=row[5].isoformat() if row[5] else "",
    )
