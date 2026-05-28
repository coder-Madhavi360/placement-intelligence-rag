from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
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

    embedding_provider: Literal["local", "sentence_transformers", "openai", "huggingface"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 32
    embedding_cache_path: Path = Path("data/vectorstores/all_minilm_l6_v2_cache.json")

    vector_store: Literal["memory", "faiss", "qdrant", "pinecone", "weaviate"] = "memory"
    vector_collection: str = "placement_intelligence"
    faiss_index_path: Path = Path("data/vectorstores/placement_intelligence.faiss")
    faiss_metadata_path: Path = Path("data/vectorstores/placement_intelligence.metadata.json")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    pinecone_api_key: str | None = None
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None

    llm_provider: Literal["stub", "openai", "anthropic", "local"] = "openai"
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("RAG_LLM_MODEL", "LLM_MODEL"),
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 500

    max_retrieval_results: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
