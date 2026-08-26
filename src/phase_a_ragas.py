from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": [
            {"rank": i + 1, "question_id": r.question_id, "distribution": r.distribution,
             "question": r.question, "avg_score": round(r.avg_score, 4),
             "worst_metric": r.worst_metric}
            for i, r in enumerate(sorted(results, key=lambda x: x.avg_score)[:10])
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved → {path}")


# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

DISTRIBUTIONS: list[Distribution] = ["factual", "multi_hop", "adversarial"]


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups: dict[str, list[dict]] = {dist: [] for dist in DISTRIBUTIONS}
    for item in test_set:
        dist = item.get("distribution")
        # Câu hỏi có distribution lạ vẫn được giữ lại thành nhóm riêng để không âm thầm mất dữ liệu.
        groups.setdefault(dist, []).append(item)
    return groups


# Chấm theo lô + ghi checkpoint sau mỗi lô. RAGAS không có cơ chế resume: một lần
# chạy 50 câu mất ~25-60 phút, và mọi gián đoạn (mất mạng, hết quota, tắt máy) đều
# xoá sạch kết quả. Đã mất 2 lần chạy vì đúng lý do này nên tách lô là bắt buộc.
RAGAS_CHUNK_SIZE = 10
RAGAS_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports", ".ragas_checkpoint.json")


def _load_checkpoint() -> dict:
    """Đọc điểm đã chấm. Hỏng file thì bỏ qua, chấm lại từ đầu còn hơn dùng số rác."""
    try:
        with open(RAGAS_CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_checkpoint(scored: dict) -> None:
    os.makedirs(os.path.dirname(RAGAS_CHECKPOINT_PATH), exist_ok=True)
    with open(RAGAS_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Gọi evaluate_ragas() từ src/m4_eval.py (Day 18) rồi ghép per-question scores
    với distribution info của test set.
    """
    if not answers:
        return []

    try:
        from src.m4_eval import evaluate_ragas
    except ImportError as e:
        print(f"⚠️  Không import được src/m4_eval.py ({e}) — đã copy từ Day 18 chưa?")
        return []

    scored = _load_checkpoint()
    todo = [a for a in answers if str(a["id"]) not in scored]
    if scored:
        print(f"Checkpoint: đã có {len(scored)}/{len(answers)} câu — chạy tiếp {len(todo)} câu.")

    print(f"Running RAGAS on {len(todo)} questions theo lô {RAGAS_CHUNK_SIZE} "
          f"(có thể mất vài chục phút)...")
    for start in range(0, len(todo), RAGAS_CHUNK_SIZE):
        chunk = todo[start:start + RAGAS_CHUNK_SIZE]
        raw = evaluate_ragas([a["question"] for a in chunk],
                             [a["answer"] for a in chunk],
                             [a["contexts"] for a in chunk],
                             [a["ground_truth"] for a in chunk])
        per_q = raw.get("per_question", [])

        # Dừng ngay khi judge trả NaN (thường là hết quota ngày hoặc không parse được
        # output). Ghi tiếp sẽ đóng băng "điểm 0 giả" vào checkpoint, và những lô sau
        # không có cách nào phân biệt được nó với điểm 0 thật.
        nan_counts = raw.get("nan_counts") or {}
        if nan_counts:
            _save_checkpoint(scored)
            raise RuntimeError(
                f"Judge trả NaN cho {nan_counts} ở lô câu "
                f"{chunk[0]['id']}-{chunk[-1]['id']}. Nguyên nhân thường gặp: judge model "
                f"hết quota ngày, hoặc model không trả đúng JSON mà RAGAS cần. "
                f"Checkpoint giữ {len(scored)} câu đã chấm — sửa RAGAS_JUDGE_POOL "
                f"trong src/m4_eval.py rồi chạy lại, các câu này sẽ không bị chấm lại.")

        if len(per_q) != len(chunk):
            print(f"⚠️  RAGAS trả về {len(per_q)}/{len(chunk)} kết quả trong lô này "
                  f"— các câu thiếu nhận score 0.0")
        for i, a in enumerate(chunk):
            pq = per_q[i] if i < len(per_q) else None
            scored[str(a["id"])] = {
                "faithfulness":      float(getattr(pq, "faithfulness", 0.0)),
                "answer_relevancy":  float(getattr(pq, "answer_relevancy", 0.0)),
                "context_precision": float(getattr(pq, "context_precision", 0.0)),
                "context_recall":    float(getattr(pq, "context_recall", 0.0)),
            }
        _save_checkpoint(scored)
        print(f"  ✓ {len(scored)}/{len(answers)} câu đã chấm (checkpoint đã ghi)",
              flush=True)

    results: list[RagasResult] = []
    for a in answers:
        m = scored.get(str(a["id"]), {})
        results.append(RagasResult(
            question_id=a["id"],
            distribution=a["distribution"],
            question=a["question"],
            answer=a["answer"],
            contexts=a["contexts"],
            ground_truth=a["ground_truth"],
            faithfulness=m.get("faithfulness", 0.0),
            answer_relevancy=m.get("answer_relevancy", 0.0),
            context_precision=m.get("context_precision", 0.0),
            context_recall=m.get("context_recall", 0.0),
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất + diagnosis từ DIAGNOSTIC_TREE.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    worst_first = sorted(results, key=lambda r: r.avg_score)[:10]

    output = []
    for i, r in enumerate(worst_first):
        diagnosis, fix = DIAGNOSTIC_TREE[r.worst_metric]
        output.append({
            "rank":          i + 1,
            "question_id":   r.question_id,
            "distribution":  r.distribution,
            "question":      r.question,
            "avg_score":     round(r.avg_score, 4),
            "worst_metric":  r.worst_metric,
            "worst_score":   round(getattr(r, r.worst_metric), 4),
            "metrics": {
                "faithfulness":      round(r.faithfulness, 4),
                "answer_relevancy":  round(r.answer_relevancy, 4),
                "context_precision": round(r.context_precision, 4),
                "context_recall":    round(r.context_recall, 4),
            },
            "diagnosis":     diagnosis,
            "suggested_fix": fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Matrix 4 metrics × 3 distributions đếm số câu hỏi mà metric đó là điểm yếu nhất.
    Từ đó suy ra distribution hay fail nhất và metric là bottleneck chủ đạo.
    """
    matrix = {metric: {dist: 0 for dist in DISTRIBUTIONS} for metric in DIAGNOSTIC_TREE}
    for r in results:
        if r.distribution in matrix[r.worst_metric]:
            matrix[r.worst_metric][r.distribution] += 1

    if not results:
        return {"matrix": matrix, "dominant_failure_distribution": None,
                "dominant_failure_metric": None,
                "insight": "Chưa có kết quả RAGAS — chạy run_ragas_50q() trước."}

    # Dominant distribution: tính theo avg_score thấp nhất, không phải theo số lượng thô
    # (factual có 20 câu còn adversarial chỉ 10 → đếm thô sẽ luôn thiên vị nhóm đông).
    per_dist_avg = {}
    for dist in DISTRIBUTIONS:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist_avg[dist] = sum(r.avg_score for r in subset) / len(subset)

    dominant_dist = min(per_dist_avg, key=per_dist_avg.get) if per_dist_avg else None
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))

    # Metric trung bình thấp nhất trên toàn bộ 50 câu — bổ sung góc nhìn cho matrix đếm.
    metric_avgs = {
        m: sum(getattr(r, m) for r in results) / len(results)
        for m in DIAGNOSTIC_TREE
    }
    weakest_metric_by_avg = min(metric_avgs, key=metric_avgs.get)

    insight = (
        f"Distribution '{dominant_dist}' yếu nhất (avg_score="
        f"{per_dist_avg.get(dominant_dist, 0):.3f}). "
        f"Metric '{dominant_metric}' là worst_metric của "
        f"{sum(matrix[dominant_metric].values())}/{len(results)} câu hỏi; "
        f"metric có điểm trung bình thấp nhất là '{weakest_metric_by_avg}' "
        f"({metric_avgs[weakest_metric_by_avg]:.3f}). "
        f"Gợi ý khắc phục: {DIAGNOSTIC_TREE[dominant_metric][1]}."
    )

    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "per_distribution_avg": {k: round(v, 4) for k, v in per_dist_avg.items()},
        "per_metric_avg": {k: round(v, 4) for k, v in metric_avgs.items()},
        "weakest_metric_by_avg": weakest_metric_by_avg,
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
    else:
        print("⚠️  No results — implement run_ragas_50q() first.")
