from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY  # noqa: F401 (giữ để tương thích import cũ)
from src.llm import chat, has_llm


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Helpers ─────────────────────────────────────────────

NL = chr(10)


def _sentences(text: str) -> list[str]:
    import re
    return [s.strip() for s in re.split(r"[.!?" + NL + r"]", text) if s.strip()]


def _parse_json(raw: str) -> dict:
    """Parse JSON từ LLM, chịu được ```json fence."""
    import json as _json, re
    if not raw:
        return {}
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return _json.loads(cleaned)
    except Exception:
        m = re.search(r"{.*}", cleaned, flags=re.DOTALL)
        if m:
            try:
                return _json.loads(m.group(0))
            except Exception:
                pass
    return {}


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk -> giảm noise.
    """
    if has_llm():
        out = chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. Chỉ trả về phần tóm tắt.",
                   text, max_tokens=150)
        if out:
            return out

    # Extractive fallback (không cần API): lấy 2 câu đầu
    sents = _sentences(text)
    return ". ".join(sents[:2]) + "." if sents else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk -> query match tốt hơn (bridge vocabulary gap).
    """
    if has_llm():
        out = chat(
            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. "
            "Mỗi câu hỏi trên 1 dòng, không đánh số, không giải thích thêm.",
            text, max_tokens=200)
        if out:
            questions = [q.strip().lstrip("0123456789.-) ") for q in out.splitlines() if q.strip()]
            if questions:
                return questions[:n_questions]

    # Extractive fallback: biến câu thành câu hỏi thô
    sents = [s for s in _sentences(text) if len(s) > 10]
    return [f"{s.rstrip('.')}?" for s in sents[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if has_llm():
        out = chat(
            "Viết đúng 1 câu tiếng Việt mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
            "Chỉ trả về 1 câu, không thêm gì khác.",
            f"Tài liệu: {document_title}" + NL + NL + "Đoạn văn:" + NL + text,
            max_tokens=100)
        if out:
            return out + NL + NL + text

    # Fallback không cần API: gắn tên tài liệu làm ngữ cảnh
    prefix = f"Trích từ tài liệu {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, category, language.
    """
    if has_llm():
        out = chat(
            'Trích xuất metadata từ đoạn văn. Chỉ trả về JSON đúng schema: '
            '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
            text, json_mode=True, max_tokens=200)
        parsed = _parse_json(out or "")
        if parsed:
            return parsed

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


_COMBINED_PROMPT = (
    "Phân tích đoạn văn tiếng Việt và chỉ trả về JSON đúng schema sau:" + NL
    + "{" + NL
    + '  "summary": "tóm tắt 2-3 câu",' + NL
    + '  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],' + NL
    + '  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì",' + NL
    + '  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}' + NL
    + "}"
)


_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           ".enrich_cache.json")
_CACHE: dict | None = None


def _cache() -> dict:
    """Cache enrichment theo hash(text+source).

    Enrich 1 lần tốn 1 API call/chunk; free tier Gemini giới hạn 15 req/phút nên
    chạy lại pipeline mà không cache sẽ tốn thêm ~5 phút và ăn vào quota ngày.
    """
    global _CACHE
    if _CACHE is None:
        import json as _json
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _CACHE = _json.load(f)
        except Exception:
            _CACHE = {}
    return _CACHE


def _cache_save() -> None:
    import json as _json
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            _json.dump(_cache(), f, ensure_ascii=False)
    except Exception as e:
        print(f"  Không ghi được enrich cache: {e}")


def _cache_key(text: str, source: str) -> str:
    import hashlib
    from config import GEMINI_CHAT_MODEL
    raw = f"{GEMINI_CHAT_MODEL}|{source}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    Cost optimization: 1 API call thay vì 4 calls riêng lẻ
    (giảm ~75% số request + latency, quan trọng khi enrich hàng nghìn chunk).
    """
    key = _cache_key(text, source)
    if key in _cache():
        return _cache()[key]

    if has_llm():
        parsed = _parse_json(chat(
            _COMBINED_PROMPT,
            f"Tài liệu: {source}" + NL + NL + "Đoạn văn:" + NL + text,
            json_mode=True, max_tokens=500) or "")
        if parsed:
            _cache()[key] = parsed
            _cache_save()
            return parsed

    # Fallback không cần API: enrichment thuần extractive để pipeline vẫn chạy
    sents = _sentences(text)
    return {
        "summary": ". ".join(sents[:2]) + "." if sents else text,
        "questions": [f"{s.rstrip('.')}?" for s in sents[:3] if len(s) > 10],
        "context": f"Trích từ tài liệu {source}." if source else "",
        "metadata": {"topic": "general", "entities": [], "category": "policy", "language": "vi"},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
