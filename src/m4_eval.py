from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, RAGAS_MAX_WORKERS
from src.llm import ragas_backend, provider


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Mỗi metric một judge model riêng. Lý do: quota free tier của Gemini tính theo
# từng model id (15 req/phút/model), nên trải 4 metric lên 3 model sẽ nhân ba throughput
# — Phase A giảm từ ~63 phút xuống ~22 phút mà không phải nới rate limit.
#
# Quan trọng: mỗi metric vẫn dùng ĐÚNG MỘT judge cho cả 50 câu hỏi, nên so sánh
# giữa các distribution (factual / multi_hop / adversarial) trong cùng một metric
# vẫn hợp lệ. Chỉ so sánh tuyệt đối GIỮA các metric khác nhau là không cùng thang —
# nhưng điều đó vốn đã đúng kể cả khi dùng chung một judge.
#
# Chia tải theo số call thực tế mỗi metric sinh ra:
#   context_precision ~3 call/câu (1 call/context)  → nặng nhất
#   faithfulness      ~2 call/câu (tách statement + NLI)
#   context_recall / answer_relevancy ~1 call/câu
# CẢNH BÁO đã gặp thật: gemini-3-flash-preview chỉ có 20 request/NGÀY
# (GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20) — cạn sau vài câu,
# metric gán vào nó retry tới hết lượt rồi trả NaN → bị quy về 0.0, tức là điểm 0 GIẢ.
# Chỉ dùng các model đã kiểm chứng còn quota ngày rộng: 3 biến thể flash-lite.
# Quota ngày đã cạn trong lúc chạy lab (mỗi model 500 req/ngày, free tier):
#   gemini-3.1-flash-lite, gemini-3.1-flash-lite-preview → hết
#   gemini-3-flash-preview → chỉ 20 req/ngày, hết sau vài câu
#   gemini-3.5-flash → cũng chỉ 20 req/ngày + 5 req/phút, không dùng làm judge được
# Gemma (gemma-4-31b-it / 26b) tuy còn quota nhưng KHÔNG dùng được làm RAGAS judge:
# không trả đúng JSON schema mà answer_relevancy cần → "Failed to parse output" → NaN.
#   gemini-3.6-flash → cũng 20 req/ngày
#   gemini-pro-latest, gemini-3.1-pro-preview → limit 0, free tier không có
# Cuối cùng chỉ còn 2 bucket sống, chia theo số call để không bucket nào cạn trước:
RAGAS_JUDGE_POOL: dict[str, str] = {
    "context_precision": "gemini-3.5-flash-lite",     # ~3 call/câu ─┐ 200 call/50 câu
    "context_recall":    "gemini-3.5-flash-lite",     # ~1 call/câu ─┘
    "faithfulness":      "gemini-flash-lite-latest",  # ~2 call/câu ─┐ 150 call/50 câu
    "answer_relevancy":  "gemini-flash-lite-latest",  # ~1 call/câu ─┘
}


def _assign_judge_pool(metrics: list) -> None:
    """Gán metric.llm theo RAGAS_JUDGE_POOL. Metric nào không có trong pool thì để
    nguyên (RAGAS sẽ dùng judge chung truyền qua evaluate(llm=...)).

    RAGAS 0.1.x chỉ ghi đè metric.llm khi nó đang là None, nên set trước là an toàn.
    """
    if provider() != "gemini":
        return
    for metric in metrics:
        model = RAGAS_JUDGE_POOL.get(getattr(metric, "name", ""))
        if not model:
            continue
        try:
            from ragas.llms import LangchainLLMWrapper
            llm, _ = ragas_backend(judge_model=model)
            metric.llm = LangchainLLMWrapper(llm)
        except Exception as e:
            print(f"  ⚠️  Không gán được judge {model} cho {metric.name} ({e}) "
                  f"— dùng judge mặc định.")


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation (4 metrics).

    Cần OPENAI_API_KEY (RAGAS dùng LLM làm judge) và Python 3.11+ (asyncio).
    Bọc try/except để pipeline vẫn chạy end-to-end khi thiếu key.
    """
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    if provider() == "none":
        print("  ⚠️  Không có GOOGLE_API_KEY/OPENAI_API_KEY — bỏ qua RAGAS (scores = 0).")
        return zeros

    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Judge LLM: Gemini nếu có GOOGLE_API_KEY, ngược lại default OpenAI của RAGAS.
        judge_llm, judge_emb = ragas_backend()
        kwargs = {}
        if judge_llm is not None:
            kwargs["llm"] = judge_llm
            kwargs["embeddings"] = judge_emb
        try:
            from ragas.run_config import RunConfig
            kwargs["run_config"] = RunConfig(max_workers=RAGAS_MAX_WORKERS, timeout=300)
        except Exception:
            pass

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        _assign_judge_pool(metrics)

        result = evaluate(dataset, metrics=metrics, **kwargs)
        df = result.to_pandas()

        # Đếm NaN thay vì im lặng quy về 0.0. RAGAS trả NaN khi judge hết quota hoặc
        # không parse được output — quy thẳng về 0.0 sẽ tạo "điểm 0 giả" trông y hệt
        # một câu trả lời thực sự tệ, và báo cáo vẫn chạy trơn tru với số liệu sai.
        nan_counts: dict[str, int] = {}

        def _f(row, key):
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                nan_counts[key] = nan_counts.get(key, 0) + 1
                return 0.0
            if value != value:  # NaN
                nan_counts[key] = nan_counts.get(key, 0) + 1
                return 0.0
            return value

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=_f(row, "faithfulness"),
                answer_relevancy=_f(row, "answer_relevancy"),
                context_precision=_f(row, "context_precision"),
                context_recall=_f(row, "context_recall"),
            )
            for _, row in df.iterrows()
        ]

        n = max(len(per_question), 1)
        return {
            "nan_counts": nan_counts,
            "faithfulness": sum(r.faithfulness for r in per_question) / n,
            "answer_relevancy": sum(r.answer_relevancy for r in per_question) / n,
            "context_precision": sum(r.context_precision for r in per_question) / n,
            "context_recall": sum(r.context_recall for r in per_question) / n,
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  RAGAS evaluation failed: {e}")
        return zeros


# Diagnostic Tree: metric yếu nhất -> nguyên nhân gốc -> cách sửa
DIAGNOSTIC_TREE = {
    "faithfulness": (
        "LLM hallucinating - câu trả lời chứa thông tin không có trong context",
        "Siết prompt ('CHỈ dùng context'), giảm temperature, thêm citation bắt buộc",
    ),
    "context_recall": (
        "Missing relevant chunks - retrieval bỏ sót thông tin cần thiết",
        "Tăng top_k, cải thiện chunking (hierarchical/structure), thêm BM25 vào hybrid",
    ),
    "context_precision": (
        "Too many irrelevant chunks - context bị nhiễu, chunk đúng xếp hạng thấp",
        "Thêm/siết reranking (cross-encoder), metadata filter theo version, giảm chunk size",
    ),
    "answer_relevancy": (
        "Answer doesn't match question - trả lời lạc đề hoặc quá chung chung",
        "Cải thiện prompt template, yêu cầu trả lời trực tiếp câu hỏi, thêm few-shot",
    ),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=lambda m: metrics[m])
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        scored.append({
            "question": r.question,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
            "answer": r.answer[:300],
            "ground_truth": r.ground_truth[:300],
        })

    scored.sort(key=lambda d: d["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
