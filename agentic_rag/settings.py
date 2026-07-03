import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

load_dotenv(REPO_ROOT / ".env")
load_dotenv(PACKAGE_DIR / ".env")


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(items) or default


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str | None
    pinecone_index_name: str
    pinecone_index_host: str | None
    pinecone_namespace: str
    pinecone_embed_model: str
    pinecone_rerank_model: str
    local_embed_model: str
    local_embed_query_instruction: str
    gemini_api_key: str | None
    gemini_model: str
    hyde_enabled: bool
    initial_top_k: int
    rerank_input_k: int
    final_top_k: int
    answer_context_chars: int
    cors_origins: tuple[str, ...]
    rrf_k: int
    wikipedia_enabled: bool
    agentic_max_plan_steps: int
    validation_enabled: bool
    validation_query_limit: int


def load_settings() -> Settings:
    return Settings(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "ald-llamamat"),
        pinecone_index_host=os.getenv("PINECONE_INDEX_HOST"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "__default__"),
        pinecone_embed_model=os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2"),
        pinecone_rerank_model=os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3"),
        local_embed_model=os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
        local_embed_query_instruction=os.getenv(
            "LOCAL_EMBED_QUERY_INSTRUCTION",
            "Represent this sentence for searching relevant passages: ",
        ),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        hyde_enabled=_parse_bool(os.getenv("HYDE_ENABLED"), True),
        initial_top_k=_parse_int(os.getenv("RAG_INITIAL_TOP_K"), 12),
        rerank_input_k=_parse_int(os.getenv("RAG_RERANK_INPUT_K"), 10),
        final_top_k=_parse_int(os.getenv("RAG_FINAL_TOP_K"), 5),
        answer_context_chars=_parse_int(os.getenv("RAG_ANSWER_CONTEXT_CHARS"), 12000),
        cors_origins=_parse_csv(
            os.getenv("RAG_CORS_ORIGINS"),
            ("http://localhost:3000", "http://127.0.0.1:3000"),
        ),
        rrf_k=_parse_int(os.getenv("RAG_RRF_K"), 60),
        wikipedia_enabled=_parse_bool(os.getenv("RAG_WIKIPEDIA_ENABLED"), True),
        agentic_max_plan_steps=_parse_int(os.getenv("AGENTIC_MAX_PLAN_STEPS"), 5),
        validation_enabled=_parse_bool(os.getenv("AGENTIC_VALIDATION_ENABLED"), True),
        validation_query_limit=_parse_int(os.getenv("AGENTIC_VALIDATION_QUERY_LIMIT"), 2),
    )
