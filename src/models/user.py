from sqlalchemy import Column, Integer, String, Enum, DateTime
from sqlalchemy.sql import func
from src.db.postgres import Base
from src.models.enums import UserRoleEnum


class User(Base):
    __tablename__ = "users"

    id               = Column(Integer, primary_key=True, index=True)
    name             = Column(String(100), nullable=False)
    email            = Column(String(200), unique=True, nullable=False)
    hashed_password  = Column(String(255), nullable=False)
    role             = Column(Enum(UserRoleEnum), default=UserRoleEnum.client)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
