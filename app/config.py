from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_provider: str = "ollama"
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None

    database_path: str = "rag.db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    embedding_dim: int = 384

    retrieval_k: int = 20
    rerank_k: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
