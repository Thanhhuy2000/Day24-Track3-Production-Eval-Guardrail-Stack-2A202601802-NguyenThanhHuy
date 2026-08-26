# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Thanh Huy (2A202601802)
**Ngày:** 26/08/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~23ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL / CREDIT_CARD / IBAN detected
    │ action:   return 400 + "PII detected in query"
    │ đo được:  chặn 4/20 câu adversarial ngay tại đây, không tốn call LLM nào
    ▼ (~1207ms P95)
[NeMo Input Rail — self_check_input, gpt-4o-mini]
    │ block if: off-topic / jailbreak / prompt injection / PII request
    │ action:   return 503 + refuse message
    │ đo được:  chặn 16/20 câu adversarial còn lại, 0/5 false positive
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Hybrid Search (BM25+Dense) → M3 Cross-encoder Rerank
    │ → gemini-3.5-flash-lite
    ▼
[Output Rail — Presidio + NeMo self_check_output]
    │ flag if:  PII in response / sensitive content
    │ action:   redact <VN_CCCD>, <VN_PHONE> rồi mới trả về
    ▼
User Response
```

---

## Latency Budget

*(Đo bằng `measure_p95_latency()`, n=10 samples)*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget | Đạt? |
|---|---|---|---|---|---|
| Presidio PII | 14.9 | 23.1 | 23.1 | <10ms | ✗ (vượt 2.3×) |
| NeMo Input Rail | 619.2 | 1207.3 | 1207.3 | <300ms | ✗ (vượt 4×) |
| RAG Pipeline | — | — | — | <2000ms | không đo trong Phase C |
| Output Rail | — | — | — | <300ms | — |
| **Total Guard** | **631.5** | **1222.4** | 1222.4 | **<500ms** | ✗ (vượt 2.4×) |

**Budget OK?** [ ] Yes / [x] **No**

**Comment:** Bottleneck là **NeMo Input Rail (1207ms P95, chiếm 99% tổng)**. Nguyên
nhân mang tính kiến trúc chứ không phải cấu hình: `self_check_input` cần một vòng gọi
LLM qua mạng, và riêng độ trễ mạng tới API đã ăn hết ngân sách 500ms.

Ba hướng xử lý, theo thứ tự tôi ưu tiên:

1. **Định tuyến hai tầng (khuyến nghị).** Cho luồng thường đi qua rail rẻ trước —
   Presidio (23ms) + so khớp embedding của Colang chạy local, không gọi mạng. Chỉ
   escalate lên `self_check_input` khi input có dấu hiệu khả nghi. Phần lớn câu hỏi HR
   hợp lệ sẽ về đích trong <100ms, P95 tổng thể tụt mạnh.
2. **Chạy model check tại chỗ.** Một model nhỏ chuyên phân loại (hoặc chính
   MiniLM đang dùng cho embedding) chạy local bỏ hẳn round-trip mạng.
3. **Sửa lại ngân sách cho đúng thực tế.** 500ms là bất khả thi cho bất kỳ input rail
   nào dựa trên LLM qua API. Nếu vẫn giữ kiến trúc này thì ngân sách trung thực là
   ~1500ms, và phải ghi rõ đây là đánh đổi có chủ đích.

**Lưu ý về Presidio:** 23.1ms so với ngân sách 10ms nghe như vượt, nhưng đây là chi phí
CPU thuần, không phụ thuộc mạng, và ổn định (P95 = P99). Trong bối cảnh tổng ngân sách
thì 23ms là không đáng kể — nên nới ngân sách tầng này lên 30ms thay vì tối ưu nó.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75      # đo được 0.753 — sát ngưỡng, cần theo dõi
    MIN_AVG_SCORE: 0.65         # đo được 0.808 — còn dư địa

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%) — đo được 20/20 (100%)

- name: False-Positive Gate       # ← gate tự thêm, xem lý do bên dưới
  run: pytest tests/test_phase_c.py -k "false_positive"
  # phải = 0/5 bị chặn nhầm

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms — HIỆN ĐANG FAIL (1222ms), xem mục Latency Budget
```

**Vì sao phải có False-Positive Gate:** trong lần chạy đầu, guard đạt adversarial
20/20 (100%) *đồng thời* chặn nhầm **5/5 câu hỏi HR hợp lệ** — false positive rate
100%. Một guard chặn tất cả sẽ luôn đạt 100% ở gate adversarial. Nếu CI chỉ có gate
adversarial thì bug này **đi thẳng lên production** với bảng điểm hoàn hảo. Hai gate
phải luôn đi cùng nhau.

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| RAGAS answer_relevancy | < 0.60 | Kiểm tra tỉ lệ câu trả lời "Không tìm thấy" |
| **Tỉ lệ câu trả lời noncommittal** | **> 20%** | **Xem mục Nhận xét — chỉ báo sớm tốt nhất** |
| Adversarial block rate | < 80% | Review new attack patterns |
| **False-positive rate trên benign** | **> 5%** | **Rollback rail config ngay** |
| Guard P95 latency | > 600ms | Scale / chuyển sang định tuyến hai tầng |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | **0.808** |
| Worst metric | **answer_relevancy (0.693)** |
| Dominant failure distribution | **multi_hop (0.716)** |
| Cohen's κ | **−0.154** (poor) — sửa lệch construct thì đạt +0.545 (moderate) |
| Adversarial pass rate | **20 / 20** (100%) |
| False-positive rate (benign) | **0 / 5** (0%) |
| Guard P95 latency | **1222 ms** (vượt ngân sách 500ms) |

Chi tiết: [failure_clusters.md](../analysis/failure_clusters.md) ·
[bias_report.md](../analysis/bias_report.md) ·
[ragas_50q.json](ragas_50q.json) · [judge_results.json](judge_results.json) ·
[guard_results.json](guard_results.json)

---

## Nhận xét & Cải tiến

> **Chạy tốt:** khâu retrieval là phần vững nhất của stack — `context_precision`
> đạt 0.963 đều ở cả ba distribution, nhờ hybrid search + cross-encoder rerank của
> Day 18. Guard cũng làm đúng việc sau khi sửa: chặn 20/20 tấn công mà không chặn nhầm
> câu hỏi hợp lệ nào, và Presidio bắt được 4/20 ca ngay ở tầng rẻ nhất (23ms, không
> tốn LLM call).
>
> **Cần cải thiện:** nút thắt nằm ở **generation**, không phải retrieval. Bằng chứng
> đắt nhất: q25 và q37 có `context_precision = 1.00` **và** `context_recall = 1.00` —
> retriever lấy về đúng chunk chứa câu trả lời — mà pipeline vẫn trả "Không tìm thấy."
> Tổng cộng 9/50 câu bị RAGAS gắn cờ *noncommittal* → 0 điểm answer_relevancy. Vậy nên
> tăng top-k hay đổi embedding model sẽ **không** cải thiện được điểm số; phải sửa
> system prompt và tách bước suy luận cho câu multi-hop.
>
> **Nếu deploy production, tôi sẽ đổi ba thứ:**
>
> 1. **Chuyển input rail sang định tuyến hai tầng.** LLM check qua API không bao giờ
>    đạt được P95 500ms. Cho luồng thường đi qua Presidio + embedding matching chạy
>    local, chỉ escalate lên LLM khi khả nghi.
> 2. **Đưa false-positive gate vào CI ngang hàng với adversarial gate.** Lab này đã
>    chứng minh vì sao: một guard chặn sạch mọi thứ đạt 100% adversarial mà hoàn toàn
>    vô dụng. Chỉ đo một chiều thì không phát hiện ra.
> 3. **Không dùng pairwise-vs-ground-truth làm cổng chất lượng.** κ = −0.154 không phải
>    do judge kém mà do đo sai thứ cần đo; chuyển sang reference-based grading với
>    rubric nhị phân, và hiệu chuẩn lại κ mỗi lần đổi model judge.
>
> **Một bài học vận hành ngoài kỹ thuật:** quota của LLM provider là ràng buộc hạng
> nhất của pipeline eval, không phải chi tiết vặt. Chạy trên Gemini free tier, 5 model
> lần lượt cạn quota giữa chừng (đo được: 20 req/ngày với `gemini-3.5-flash`,
> `3.6-flash`, `3-flash-preview`; 500 req/ngày với `3.1-flash-lite`). Nguy hiểm nhất
> là RAGAS trả `NaN` khi judge hết quota, và code ban đầu quy `NaN` thành `0.0` — tạo
> ra **điểm 0 giả trông y hệt điểm 0 thật**, báo cáo vẫn chạy trơn tru với số liệu sai.
> Đã sửa: `evaluate_ragas()` đếm `NaN` và Phase A dừng hẳn kèm thông báo rõ thay vì ghi
> tiếp; đồng thời thêm checkpoint theo lô 10 câu để một lần chạy 25-60 phút không mất
> trắng khi gián đoạn. Trong production, eval pipeline **phải** fail loud khi judge
> chết — im lặng nuốt lỗi ở tầng đo lường là kiểu hỏng tốn kém nhất, vì nó làm hỏng
> chính công cụ dùng để phát hiện hỏng hóc.
