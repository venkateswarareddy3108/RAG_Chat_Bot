"""Application configuration loaded from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _split_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    frontend_origins: list[str] = field(
        default_factory=lambda: _split_origins(
            os.getenv(
                "FRONTEND_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            )
        )
    )
    upload_dir: Path = field(default_factory=lambda: BASE_DIR / "uploads")
    chroma_dir: Path = field(default_factory=lambda: BASE_DIR / "chroma_db")
    docs_meta_file: Path = field(default_factory=lambda: BASE_DIR / "docs_meta.json")
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_batch_size: int = 32
    retrieval_top_k: int = 5

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
