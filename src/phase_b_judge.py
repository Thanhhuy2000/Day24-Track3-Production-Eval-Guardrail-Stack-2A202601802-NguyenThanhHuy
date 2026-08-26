from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (OPENAI_API_KEY, GOOGLE_API_KEY, JUDGE_MODEL,
                    HUMAN_LABELS_PATH, TEST_SET_PATH)
from src.llm import chat, provider

VALID_WINNERS = {"A", "B", "tie"}
SWAP_MAP = {"A": "B", "B": "A", "tie": "tie"}


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "Bạn là chuyên gia đánh giá chất lượng câu trả lời của hệ thống RAG tra cứu "
    "chính sách nhân sự. Bạn đánh giá khách quan, KHÔNG ưu tiên câu trả lời dài hơn "
    "và KHÔNG ưu tiên câu trả lời xuất hiện trước. Chỉ trả lời bằng JSON hợp lệ."
)

PROMPT_TEMPLATE = """Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá hai câu trả lời theo 3 tiêu chí (trọng số giảm dần):
1. Độ chính xác — con số, điều kiện, phiên bản chính sách có đúng không?
2. Độ đầy đủ — có trả lời hết các phần của câu hỏi không?
3. Tính súc tích — có thừa thông tin không liên quan không?

Câu trả lời DÀI HƠN không đồng nghĩa với TỐT HƠN. Nếu hai câu tương đương, chọn "tie".

Trả lời JSON đúng schema (chỉ JSON, không thêm text):
{{"winner": "A" | "B" | "tie", "reasoning": "giải thích ngắn gọn bằng tiếng Việt",
  "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}"""


def _parse_judge_json(raw: str) -> dict:
    """Parse JSON từ LLM, chịu được ```json fences và text thừa xung quanh."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # raw_decode đọc đúng một giá trị JSON rồi dừng, nên chịu được text thừa phía sau.
    # Regex greedy \{.*\} thì không: gặp model trả hai object liền nhau (Gemma hay làm)
    # nó gộp cả hai thành một chuỗi không parse được → "Extra data: line 2 column 1".
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"\{", text)):
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("Không tìm thấy object JSON hợp lệ", text, 0)


def _clamp_score(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    neutral = {"winner": "tie", "reasoning": "", "scores": {"A": 0.0, "B": 0.0}}

    if provider() == "none":
        print("  ⚠️  Không có GEMINI_API_KEY/OPENAI_API_KEY — judge trả về 'tie'.")
        return neutral

    raw = chat(
        JUDGE_SYSTEM,
        PROMPT_TEMPLATE.format(question=question, answer_a=answer_a, answer_b=answer_b),
        json_mode=True,
        max_tokens=512,
        model=JUDGE_MODEL,
    )
    if not raw:
        return neutral

    try:
        parsed = _parse_judge_json(raw)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  Judge trả về JSON không hợp lệ ({e}) — coi như 'tie'.")
        return neutral

    winner = str(parsed.get("winner", "tie")).strip()
    if winner not in VALID_WINNERS:
        # Model đôi khi trả "Answer A" / "a" — normalize về không gian A/B/tie.
        upper = winner.upper()
        winner = "A" if upper.startswith("A") else "B" if upper.startswith("B") else "tie"

    scores = parsed.get("scores") or {}
    return {
        "winner": winner,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "scores": {"A": _clamp_score(scores.get("A")), "B": _clamp_score(scores.get("B"))},
    }


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)      # SWAP!

    # Đưa pass 2 về lại không gian A/B gốc trước khi so sánh.
    winner_pass2 = SWAP_MAP[pass2_raw["winner"]]
    scores_pass2 = {"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]}

    position_consistent = pass1["winner"] == winner_pass2
    final = pass1["winner"] if position_consistent else "tie"

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"], scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                           0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Tính tay theo κ = (p_o - p_e) / (1 - p_e), tổng quát cho nhiều hơn 2 nhãn
    (không phụ thuộc sklearn).
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError(
            f"judge_labels ({len(judge_labels)}) và human_labels ({len(human_labels)}) "
            "phải cùng độ dài"
        )
    n = len(judge_labels)
    if n == 0:
        return 0.0

    p_o = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n

    categories = set(judge_labels) | set(human_labels)
    p_e = sum(
        (judge_labels.count(c) / n) * (human_labels.count(c) / n)
        for c in categories
    )

    if abs(1.0 - p_e) < 1e-12:
        # Cả hai rater dùng đúng một nhãn cho mọi item → κ không xác định.
        # Quy ước: đồng thuận hoàn toàn → 1.0, ngược lại → 0.0.
        return 1.0 if p_o == 1.0 else 0.0

    return (p_o - p_e) / (1.0 - p_e)


def interpret_kappa(kappa: float) -> str:
    """Diễn giải κ theo thang Landis-Koch."""
    if kappa < 0:
        return "poor (tệ hơn ngẫu nhiên)"
    if kappa < 0.2:
        return "slight"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "almost perfect"


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → % case có position_consistent = False.

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Trong các case có winner rõ ràng, bao nhiêu % winner là câu dài hơn.
          Baseline ngẫu nhiên = 0.5; > 0.6 là đáng lo ngại.
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0,
                                  "total_decisive": 0},
            "interpretation": "Chưa có kết quả judge nào để phân tích.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive else 0.0

    notes = [
        f"Position bias {position_bias_rate:.0%} — "
        + ("cao, bắt buộc dùng swap-and-average và coi case mâu thuẫn là 'tie'."
           if position_bias_rate > 0.3 else
           "thấp, judge ổn định theo thứ tự trình bày.")
    ]
    if decisive == 0:
        notes.append("Không có case decisive nào nên chưa kết luận được verbosity bias.")
    else:
        notes.append(
            f"Verbosity bias {verbosity_bias:.0%} trên {decisive} case decisive — "
            + ("judge thiên vị câu trả lời dài hơn, nên siết ràng buộc 'dài không đồng "
               "nghĩa tốt' trong prompt." if verbosity_bias > 0.6 else
               "không thấy thiên vị rõ rệt theo độ dài (baseline ngẫu nhiên 50%).")
        )

    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": " ".join(notes),
    }


# ─── Chạy judge trên 10 câu có human label ────────────────────────────────────

def _ground_truth_map(path: str = TEST_SET_PATH) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        return {item["id"]: item["ground_truth"] for item in json.load(f)}


def judge_human_labeled_set(
    path: str = HUMAN_LABELS_PATH,
) -> tuple[list[JudgeResult], list[int], list[int]]:
    """Chạy swap-and-average trên 10 câu có nhãn người.

    Quy ước để so sánh được với human_label (0/1):
        A = model_answer (câu trả lời của pipeline), B = ground_truth.
        Judge thấy A thắng hoặc hoà        → model_answer đủ tốt → label 1.
        Judge thấy B (ground truth) thắng  → model_answer thiếu/sai → label 0.
    """
    with open(path, encoding="utf-8") as f:
        human_data = json.load(f)
    gt_map = _ground_truth_map()

    judge_results, judge_labels, human_labels = [], [], []
    for i, item in enumerate(human_data, 1):
        ground_truth = gt_map.get(item["question_id"], "")
        result = swap_and_average(item["question"], item["model_answer"], ground_truth)
        judge_results.append(result)
        judge_labels.append(0 if result.final_winner == "B" else 1)
        human_labels.append(item["human_label"])
        print(f"  [{i}/{len(human_data)}] q{item['question_id']}: "
              f"judge={judge_labels[-1]} human={human_labels[-1]} "
              f"(winner={result.final_winner}, consistent={result.position_consistent})",
              flush=True)

    return judge_results, judge_labels, human_labels


def save_phase_b_report(judge_results: list[JudgeResult], judge_labels: list[int],
                        human_labels: list[int], kappa: float, bias: dict,
                        path: str = "reports/judge_results.json") -> None:
    """Save Phase B report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    agreement = (
        sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / len(judge_labels)
        if judge_labels else 0.0
    )
    report = {
        "judge_model": JUDGE_MODEL,
        "provider": provider(),
        "n_judged": len(judge_results),
        "cohen_kappa": round(kappa, 4),
        "kappa_interpretation": interpret_kappa(kappa),
        "agreement_rate": round(agreement, 4),
        "judge_labels": judge_labels,
        "human_labels": human_labels,
        "bias_report": bias,
        "details": [asdict(r) for r in judge_results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Judge provider={provider()} model={JUDGE_MODEL}\n")

    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge (demo)...")
    demo = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {demo.winner_pass1}")
    print(f"  Pass 2 winner: {demo.winner_pass2}")
    print(f"  Final:         {demo.final_winner}")
    print(f"  Position consistent: {demo.position_consistent}")

    # --- Judge 10 câu có nhãn người → Cohen's κ ---
    print("\nJudging 10 human-labeled questions...")
    results, judge_labels, human_labels = judge_human_labeled_set()

    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"\nCohen's κ = {kappa:.3f} → {interpret_kappa(kappa)}")

    # --- Bias report ---
    bias = bias_report(results)
    print(f"Position bias: {bias['position_bias_rate']:.0%} "
          f"({bias['position_bias_count']}/{bias['total_judged']})")
    print(f"Verbosity bias: {bias['verbosity_bias']:.0%}")
    print(f"→ {bias['interpretation']}")

    save_phase_b_report(results, judge_labels, human_labels, kappa, bias)
