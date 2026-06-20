from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    POSTGRES_USER: str = "forzy"
    POSTGRES_PASSWORD: str = "forzy123"
    POSTGRES_DB: str = "forzy_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # JWT
    SECRET_KEY: str = "troque_por_uma_chave_secreta_forte_aqui"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ANEEL
    ANEEL_RESOURCE_ID: str = "476d0490-a225-4de7-89a8-bb7a189f0868"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            f"?sslmode=require"
        )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
