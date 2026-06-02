from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from version import __version__


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    app_version: str = __version__

    log_level: str = "INFO"
    log_json: bool = False

    embedding_provider: Literal["local", "sentence_transformers", "openai", "huggingface"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 32
    embedding_cache_path: Path = Path("data/vectorstores/all_minilm_l6_v2_cache.json")

    chunk_max_tokens: int = Field(default=260, ge=80, le=2000)
    chunk_overlap_tokens: int = Field(default=40, ge=0, le=500)
    chunk_min_tokens: int = Field(default=20, ge=1, le=500)

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

    max_retrieval_results: int = Field(default=5, ge=1, le=25)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError("RAG_CHUNK_OVERLAP_TOKENS must be smaller than RAG_CHUNK_MAX_TOKENS")
        if self.chunk_min_tokens > self.chunk_max_tokens:
            raise ValueError("RAG_CHUNK_MIN_TOKENS must be less than or equal to RAG_CHUNK_MAX_TOKENS")
        return self

    def resolve_path(self, path: Path) -> Path:
        """Resolve runtime paths consistently from cwd first, then repo root."""
        if path.is_absolute():
            return path

        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return cwd_path

        return (PROJECT_ROOT / path).resolve()

    def validate_startup(self) -> list[str]:
        """Return non-fatal startup warnings for missing runtime prerequisites."""
        warnings: list[str] = []

        if self.llm_provider == "openai" and not self.openai_api_key:
            warnings.append("OpenAI API key is not configured; answer generation will use grounded fallback mode.")

        if self.vector_store == "faiss":
            index_path = self.resolve_path(self.faiss_index_path)
            metadata_path = self.resolve_path(self.faiss_metadata_path)
            if not index_path.exists() or not metadata_path.exists():
                warnings.append(
                    "FAISS index or metadata file is missing; query endpoints will return 503 until the index is built. "
                    f"index_path={index_path} metadata_path={metadata_path}"
                )

        return warnings


@lru_cache
def get_settings() -> Settings:
    return Settings()



