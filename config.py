"""Shared configuration for Lab 24: Eval + Guardrail Stack."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
# Provider dùng cho generation, RAGAS judge (Phase A), LLM-as-Judge (Phase B) và
# NeMo Guardrails (Phase C).
def _key(*names: str) -> str:
    """Đọc API key, coi placeholder trong .env.example là chưa cấu hình.

    Không lọc thì "sk-..." vẫn là chuỗi khác rỗng → code tưởng đã có key, chạy tới
    lúc gọi API mới báo 401 ở tận đáy stack trace.
    """
    for name in names:
        value = os.getenv(name, "").strip()
        if value and not value.endswith("...") and value not in {"sk-", "AIza"}:
            return value
    return ""


OPENAI_API_KEY = _key("OPENAI_API_KEY")
GOOGLE_API_KEY = _key("GOOGLE_API_KEY", "GEMINI_API_KEY")

# Chọn provider TƯỜNG MINH: "openai" | "gemini" | "auto".
# Trước đây provider() chỉ xét "có GOOGLE_API_KEY thì dùng Gemini", nên khi có cả hai
# key thì không thể chuyển sang OpenAI mà không xoá key Gemini — trong khi Phase C
# (NeMo) vẫn cần key Gemini để chạy Gemma. Tách biến riêng để chọn độc lập với key.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

# --- LLM models (Day 18 llm.py) ---
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
# Timeout mỗi request LLM (giây). google-generativeai KHÔNG đặt timeout mặc định:
# một kết nối treo sẽ block vô hạn và cả pipeline đứng im, không log gì.
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))
LLM_RPM = int(os.getenv("LLM_RPM", "14"))          # free tier Gemini = 15 req/phút
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "2"))

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab24_production"
NAIVE_COLLECTION = "lab24_naive"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set_50q.json")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(os.path.dirname(__file__), "adversarial_set_20.json")
GUARDRAILS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

# --- LLM Judge (Phase B) ---
# Judge dùng model KHÁC generator để tránh self-preference bias.
def _active_provider() -> str:
    """Provider thực sự sẽ dùng, sau khi xét LLM_PROVIDER và key hiện có."""
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        return "openai"
    if LLM_PROVIDER == "gemini" and GOOGLE_API_KEY:
        return "gemini"
    if LLM_PROVIDER in ("openai", "gemini"):
        # Chọn tường minh nhưng thiếu key → báo sớm, đừng im lặng rơi về provider khác
        # rồi để người dùng tưởng đang chạy trên model mình đã chọn.
        raise RuntimeError(
            f"LLM_PROVIDER={LLM_PROVIDER} nhưng thiếu "
            f"{'OPENAI_API_KEY' if LLM_PROVIDER == 'openai' else 'GOOGLE_API_KEY'} trong .env")
    return "gemini" if GOOGLE_API_KEY else ("openai" if OPENAI_API_KEY else "none")


ACTIVE_PROVIDER = _active_provider()
JUDGE_MODEL = GEMINI_JUDGE_MODEL if ACTIVE_PROVIDER == "gemini" else OPENAI_JUDGE_MODEL

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # target: full guard stack P95 < 500ms
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer
