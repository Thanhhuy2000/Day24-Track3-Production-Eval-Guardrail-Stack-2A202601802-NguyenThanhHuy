from __future__ import annotations

"""Production RAG Pipeline — ghép M1 + M5 + M2 + M3 + LLM + M4.

Kiến trúc chạy theo PHA để tiết kiệm RAM (máy lab chỉ có ~2GB trống):
    Pha A: retrieve toàn bộ query (bge-m3 ~1.5GB trong RAM) → unload encoder
    Pha B: rerank toàn bộ query (bge-reranker ~1.5GB trong RAM) → unload reranker
    Pha C: sinh câu trả lời + RAGAS (chỉ gọi API, không dùng model local)
Nếu load cả 2 model cùng lúc → Windows fatal exception: access violation.
"""

import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from src.llm import answer_from_context, provider
from config import RERANK_TOP_K, HYBRID_TOP_K

# Latency breakdown (bonus): tổng thời gian từng bước, ms
LATENCY: dict[str, float] = {}


def _mark(step: str, t0: float) -> float:
    elapsed = time.time() - t0
    LATENCY[step] = round(elapsed * 1000, 1)
    return elapsed


def build_pipeline():
    """Build production RAG pipeline: chunk → enrich → index."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print(f"LLM provider: {provider()}")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1 — hierarchical: retrieve child, giữ parent_id để mở rộng context)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text,
                               "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({_mark('1_chunking', t0):.1f}s)", flush=True)

    # Step 2: Enrichment (M5 — combined mode: 1 API call/chunk)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({_mark('2_enrichment', t0):.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2 — BM25 tiếng Việt + Dense bge-m3)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({_mark('3_indexing', t0):.1f}s)", flush=True)

    # Step 4: Reranker (M3 — chưa load model ở đây để tiết kiệm RAM, load lazy khi rerank)
    t0 = time.time()
    print("\n[4/4] Preparing reranker (lazy load)...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({_mark('4_reranker_init', t0):.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query end-to-end: hybrid search → rerank → LLM answer.

    Dùng cho tra cứu lẻ. Khi eval cả test set, evaluate_pipeline() chạy theo pha
    để không giữ 2 model trong RAM cùng lúc.
    """
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:RERANK_TOP_K]]
    return answer_from_context(query, contexts), contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set — 3 pha để giữ RAM thấp."""
    test_set = load_test_set()
    n = len(test_set)
    questions = [item["question"] for item in test_set]
    ground_truths = [item["ground_truth"] for item in test_set]

    # ── Pha A: Hybrid retrieval (bge-m3 đang trong RAM) ──
    print(f"\n[Eval A] Hybrid retrieval cho {n} câu hỏi...", flush=True)
    t0 = time.time()
    retrieved: list[list[dict]] = []
    for i, q in enumerate(questions):
        results = search.search(q, top_k=HYBRID_TOP_K)
        retrieved.append([{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results])
        print(f"  [{i+1}/{n}] {q[:50]}... → {len(results)} docs", flush=True)
    _mark("5_retrieval_total", t0)
    LATENCY["5_retrieval_per_query"] = round(LATENCY["5_retrieval_total"] / max(n, 1), 1)

    search.dense.unload()  # giải phóng bge-m3 trước khi load cross-encoder
    print("  ✓ Đã unload embedding model để nhường RAM cho reranker", flush=True)

    # ── Pha B: Reranking (cross-encoder trong RAM) ──
    print(f"\n[Eval B] Reranking top-{HYBRID_TOP_K} → top-{RERANK_TOP_K}...", flush=True)
    t0 = time.time()
    all_contexts: list[list[str]] = []
    for i, (q, docs) in enumerate(zip(questions, retrieved)):
        reranked = reranker.rerank(q, docs, top_k=RERANK_TOP_K)
        contexts = [r.text for r in reranked] if reranked else [d["text"] for d in docs[:RERANK_TOP_K]]
        all_contexts.append(contexts or ["Không có context."])
        print(f"  [{i+1}/{n}] reranked", flush=True)
    _mark("6_rerank_total", t0)
    LATENCY["6_rerank_per_query"] = round(LATENCY["6_rerank_total"] / max(n, 1), 1)

    reranker.unload()  # giải phóng cross-encoder trước khi gọi LLM/RAGAS
    print("  ✓ Đã unload reranker", flush=True)

    # ── Pha C: Generation + RAGAS (chỉ gọi API) ──
    print(f"\n[Eval C] Sinh câu trả lời ({provider()})...", flush=True)
    t0 = time.time()
    answers = []
    for i, (q, contexts) in enumerate(zip(questions, all_contexts)):
        answers.append(answer_from_context(q, contexts))
        print(f"  [{i+1}/{n}] answered", flush=True)
    _mark("7_generation_total", t0)
    LATENCY["7_generation_per_query"] = round(LATENCY["7_generation_total"] / max(n, 1), 1)

    # Checkpoint: retrieval + rerank + generation tốn ~15 phút và quota API.
    # Lưu lại để có thể chạy lại RAGAS riêng (python src/pipeline.py --eval-only)
    # khi RAGAS bị gián đoạn vì rate limit.
    save_predictions(questions, answers, all_contexts, ground_truths)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {n} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({_mark('8_ragas', t0):.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures)
    save_latency_report(n)
    return results


def save_latency_report(n_queries: int, path: str = "latency_report.json"):
    """Latency breakdown từng bước (bonus +2)."""
    report = {"n_queries": n_queries, "steps_ms": LATENCY}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    print(f"{'Step':<28} {'Total (ms)':>12} {'Per query (ms)':>16}")
    print("-" * 58)
    for step in sorted(LATENCY):
        if step.endswith("_per_query"):
            continue
        total = LATENCY[step]
        per_q = LATENCY.get(step.replace("_total", "_per_query"))
        per_q_str = f"{per_q:.1f}" if per_q is not None else "-"
        print(f"{step:<28} {total:>12.1f} {per_q_str:>16}")
    print(f"\nLatency report saved to {path}")


PREDICTIONS_PATH = "pipeline_predictions.json"


def save_predictions(questions, answers, contexts, ground_truths, path: str = PREDICTIONS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "answers": answers,
                   "contexts": contexts, "ground_truths": ground_truths},
                  f, ensure_ascii=False, indent=2)
    print(f"  ✓ Predictions checkpoint → {path}", flush=True)


def regenerate_answers(path: str = PREDICTIONS_PATH):
    """Sinh lại câu trả lời từ context đã lưu trong checkpoint.

    Dùng khi lần chạy trước bị 429 giữa đường và rơi vào fallback `answer = contexts[0]`
    (fallback này làm faithfulness bị thổi lên giả vì answer trùng context).
    Không phải index + rerank lại.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions, contexts = data["questions"], data["contexts"]
    print(f"[Regen] Sinh lại {len(questions)} câu trả lời ({provider()})...", flush=True)

    answers, fallbacks = [], 0
    for i, (q, ctx) in enumerate(zip(questions, contexts)):
        answer = answer_from_context(q, ctx)
        if ctx and answer.strip() == ctx[0].strip():
            fallbacks += 1
        answers.append(answer)
        print(f"  [{i+1}/{len(questions)}] {'FALLBACK' if ctx and answer.strip() == ctx[0].strip() else 'ok'}", flush=True)

    print(f"  ✓ Xong — {fallbacks}/{len(questions)} câu vẫn là fallback", flush=True)
    save_predictions(questions, answers, contexts, data["ground_truths"], path=path)
    return fallbacks


def evaluate_from_predictions(path: str = PREDICTIONS_PATH):
    """Chỉ chạy lại RAGAS từ checkpoint (không retrieval/rerank/generate lại)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"[Eval-only] RAGAS từ {path} ({len(data['questions'])} câu hỏi)...", flush=True)
    t0 = time.time()
    results = evaluate_ragas(data["questions"], data["answers"],
                             data["contexts"], data["ground_truths"])
    print(f"  ✓ RAGAS done ({_mark('8_ragas', t0):.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    if "--regen" in sys.argv:
        regenerate_answers()
        evaluate_from_predictions()
    elif "--eval-only" in sys.argv:
        evaluate_from_predictions()
    else:
        search, reranker = build_pipeline()
        evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
