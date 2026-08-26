"""
Setup script: chạy Day 18 pipeline trên 50 câu hỏi → lưu answers_50q.json

Chạy TRƯỚC khi bắt đầu Phase A:
    python setup_answers.py

Yêu cầu:
    1. Đã copy src/ từ Day 18 (m1-m5, pipeline.py, llm.py) vào thư mục này
    2. docker compose up -d  (Qdrant đang chạy trên port 6333)
    3. .env có GEMINI_API_KEY hoặc OPENAI_API_KEY

Ghi chú (khác bản scaffold gốc):
    - Sinh câu trả lời qua src/llm.py :: answer_from_context() thay vì gọi thẳng
      OpenAI, để chạy được với provider nào cũng được (Gemini/OpenAI) — cùng một
      generator với Day 18 pipeline.
    - Chạy theo 3 PHA (retrieve → unload → rerank → unload → generate) như
      src/pipeline.py :: evaluate_pipeline(), vì giữ bge-m3 và cross-encoder trong
      RAM cùng lúc gây access violation trên máy ít RAM.
    - Checkpoint sau mỗi pha vào .setup_checkpoint.json: retrieval + rerank tốn
      ~15 phút, không nên chạy lại từ đầu khi LLM dính rate limit.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH = ".setup_checkpoint.json"


def check_day18_files() -> bool:
    required = [
        "src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
        "src/m4_eval.py",     "src/m5_enrichment.py", "src/pipeline.py",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n❌ Thiếu files từ Day 18. Copy chúng vào src/ trước:\n")
        for f in missing:
            print(f"   cp <Day18>/src/{os.path.basename(f)} src/")
        return False
    print(f"✓ Day 18 source files: {len(required)}/{len(required)} found")
    return True


def build_pipeline():
    """Chunk → enrich → index → chuẩn bị reranker (dùng lại build_pipeline của Day 18)."""
    from src.pipeline import build_pipeline as day18_build
    return day18_build()


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(data: dict) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main():
    print("=" * 60)
    print("LAB 24 SETUP — Generating answers for 50 questions")
    print("=" * 60)

    if not check_day18_files():
        sys.exit(1)

    from config import HYBRID_TOP_K, RERANK_TOP_K
    from src.llm import answer_from_context, provider

    with open("test_set_50q.json", encoding="utf-8") as f:
        test_set = json.load(f)
    n = len(test_set)
    print(f"✓ Loaded {n} questions (factual/multi_hop/adversarial)")
    print(f"✓ LLM provider: {provider()}")

    questions = [item["question"] for item in test_set]
    checkpoint = load_checkpoint()

    # ── Pha A+B: retrieval + rerank (cần model local, tốn RAM) ────────────────
    if len(checkpoint.get("contexts", [])) == n:
        all_contexts = checkpoint["contexts"]
        print(f"\n✓ Dùng lại contexts từ {CHECKPOINT_PATH} (bỏ qua retrieval + rerank)")
    else:
        try:
            search, reranker = build_pipeline()
        except ImportError as e:
            print(f"\n❌ Import error: {e}")
            print("→ Đã copy src/ từ Day 18 và pip install -r requirements.txt chưa?")
            sys.exit(1)

        print(f"\n[Pha A] Hybrid retrieval cho {n} câu hỏi...", flush=True)
        t0 = time.time()
        retrieved = []
        for i, q in enumerate(questions):
            results = search.search(q, top_k=HYBRID_TOP_K)
            retrieved.append([{"text": r.text, "score": r.score, "metadata": r.metadata}
                              for r in results])
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n}] retrieved ({time.time()-t0:.0f}s)", flush=True)

        search.dense.unload()   # nhường RAM cho cross-encoder
        print("  ✓ Đã unload embedding model", flush=True)

        print(f"\n[Pha B] Reranking top-{HYBRID_TOP_K} → top-{RERANK_TOP_K}...", flush=True)
        t0 = time.time()
        all_contexts = []
        for i, (q, docs) in enumerate(zip(questions, retrieved)):
            reranked = reranker.rerank(q, docs, top_k=RERANK_TOP_K)
            contexts = ([r.text for r in reranked] if reranked
                        else [d["text"] for d in docs[:RERANK_TOP_K]])
            all_contexts.append(contexts or ["Không có context."])
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n}] reranked ({time.time()-t0:.0f}s)", flush=True)

        reranker.unload()
        print("  ✓ Đã unload reranker", flush=True)

        checkpoint["contexts"] = all_contexts
        save_checkpoint(checkpoint)

    # ── Pha C: generation (chỉ gọi API) ───────────────────────────────────────
    print(f"\n[Pha C] Sinh câu trả lời bằng {provider()}...", flush=True)
    t0 = time.time()
    answers = []
    for i, (item, contexts) in enumerate(zip(test_set, all_contexts)):
        answer = answer_from_context(item["question"], contexts)
        answers.append({
            "id":           item["id"],
            "distribution": item["distribution"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] answered ({time.time()-t0:.0f}s)", flush=True)

    with open("answers_50q.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(answers)} answers → answers_50q.json")
    print(f"  Total generation time: {time.time()-t0:.1f}s")
    print("\n→ Bây giờ bắt đầu Phase A:")
    print("     python src/phase_a_ragas.py")


if __name__ == "__main__":
    main()
