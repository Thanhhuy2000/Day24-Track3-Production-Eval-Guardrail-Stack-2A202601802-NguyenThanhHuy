from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

_ST_MODEL = None


def _get_sentence_model(name: str = "all-MiniLM-L6-v2"):
    """Lazy-load + cache embedding model (tránh load lại mỗi lần gọi)."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(name)
    return _ST_MODEL


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0],
                      metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    model = _get_sentence_model()
    embeddings = model.encode(sentences)

    def cosine_sim(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([sentences[i]])          # đổi chủ đề → chunk mới
        else:
            groups[-1].append(sentences[i])        # cùng chủ đề → gộp

    return [
        Chunk(text=" ".join(g),
              metadata={**metadata, "strategy": "semantic", "chunk_index": i,
                        "n_sentences": len(g)})
        for i, g in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _pack(units: list[str], max_size: int, sep: str = "\n\n") -> list[str]:
    """Gộp các đơn vị text thành block ≤ max_size chars (giữ nguyên thứ tự)."""
    blocks, current = [], ""
    for u in units:
        if current and len(current) + len(sep) + len(u) > max_size:
            blocks.append(current)
            current = u
        else:
            current = f"{current}{sep}{u}" if current else u
    if current.strip():
        blocks.append(current)
    return blocks


def _split_long(unit: str, max_size: int) -> list[str]:
    """Cắt cứng 1 đơn vị dài hơn max_size (fallback khi 1 câu/đoạn quá dài)."""
    return [unit[i:i + max_size] for i in range(0, len(unit), max_size)] or [unit]


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    # Paragraph dài hơn parent_size → cắt nhỏ trước khi đóng gói
    units: list[str] = []
    for p in paragraphs:
        units.extend(_split_long(p, parent_size) if len(p) > parent_size else [p])

    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_text in _pack(units, parent_size):
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                      "strategy": "hierarchical", "chunk_index": len(parents)},
        ))

        # Child: cắt parent theo câu rồi đóng gói lại ≤ child_size
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', parent_text) if s.strip()]
        child_units: list[str] = []
        for s in sentences:
            child_units.extend(_split_long(s, child_size) if len(s) > child_size else [s])

        for j, child_text in enumerate(_pack(child_units, child_size, sep=" ")):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "strategy": "hierarchical",
                          "parent_id": pid, "child_index": j},
                parent_id=pid,
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    parts = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    def _flush():
        body = f"{current_header}\n\n{current_content.strip()}" if current_header else current_content.strip()
        if body.strip():
            chunks.append(Chunk(
                text=body.strip(),
                metadata={**metadata, "section": current_header.lstrip("# ").strip() or "intro",
                          "header_level": current_header.count("#") or 0,
                          "strategy": "structure", "chunk_index": len(chunks)},
            ))

    for part in parts:
        if not part.strip():
            continue
        if re.match(r'^#{1,3}\s+.+$', part.strip()):
            _flush()                       # đóng section trước đó
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
    _flush()

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
