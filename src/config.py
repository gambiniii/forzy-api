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

    # InfluxDB
    INFLUX_URL: str = "http://localhost:8086"
    INFLUX_TOKEN: str = "forzy-token-local"
    INFLUX_ORG: str = "forzy"
    INFLUX_BUCKET: str = "sensor_readings"

    # LLM / RAG (OpenRouter + Gemini)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "google/gemini-2.5-flash"

    # Forzy Sensor API
    FORZY_API_BASE_URL: str = "http://localhost:8000"
    FORZY_SENSOR_URL: str = "https://reseller-prescribed-facing-dept.trycloudflare.com"

    # RAG
    CHROMA_PERSIST_DIR: str = "rag_module/vectorstore"
    DOCUMENTS_DIR: str = "rag_module/documents"

    # Caminho para repositório externo de ML (opcional)
    ML_MODULE_PATH: str = ""

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
