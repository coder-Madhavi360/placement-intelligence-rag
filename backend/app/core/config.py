from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven application settings.

    Keep credentials out of code. Values are loaded from environment variables
    and optionally from a local .env file during development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    app_name: str = "Placement Intelligence RAG"
    environment: Literal["local", "development", "staging", "production"] = "local"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    log_level: str = "INFO"
    log_json: bool = False

    embedding_provider: Literal["local", "openai", "huggingface"] = "local"
    embedding_model: str = "local-hash-embedding"
    embedding_dimensions: int = 384

    vector_store: Literal["memory", "qdrant", "pinecone", "weaviate"] = "memory"
    vector_collection: str = "placement_intelligence"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    pinecone_api_key: str | None = None
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None

    llm_provider: Literal["stub", "openai", "anthropic", "local"] = "stub"
    llm_model: str = "stub"
    openai_api_key: str | None = None

    max_retrieval_results: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
