# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Thanh Huy (2A202601802)
**Ngày:** 26/08/2026
**Judge model:** `gpt-4o-mini` (RAGAS 0.1.22) — khác nhà cung cấp với generator
`gemini-3.5-flash-lite` nên không có self-preference bias.

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual (20) | multi_hop (20) | adversarial (10) | **Toàn bộ 50 câu** |
|---|---|---|---|---|
| faithfulness | 0.933 | **0.521** | 0.855 | 0.753 |
| answer_relevancy | 0.795 | 0.583 | 0.708 | **0.693** |
| context_precision | 0.967 | 0.967 | 0.950 | **0.963** |
| context_recall | 0.908 | 0.792 | 0.717 | 0.823 |
| **avg_score** | **0.901** | **0.716** | 0.808 | 0.808 |

Đọc bảng theo cột: pipeline mạnh ở `factual` (0.901), tụt hẳn ở `multi_hop` (0.716).
Đọc theo hàng: `context_precision` gần như hoàn hảo (0.963) ở **cả ba** distribution,
trong khi `answer_relevancy` thấp nhất (0.693). Đây là chỉ dấu quan trọng nhất của
toàn bộ bài: khâu tìm kiếm không phải là điểm yếu — khâu sinh câu trả lời mới là.

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question (rút gọn) | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | q39 — So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 | 0.125 | faithfulness |
| 2 | adversarial | q44 — Bao lâu phải đổi mật khẩu một lần? | 0.375 | faithfulness |
| 3 | multi_hop | q33 — Manager 12 năm: tổng phụ cấp + số ngày phép | 0.375 | faithfulness |
| 4 | multi_hop | q35 — Junior P1 lương 12tr vừa thử việc: nhận lương bao nhiêu | 0.375 | faithfulness |
| 5 | multi_hop | q31 — Công tác 2 ngày, khách sạn 1.5tr/đêm: thanh toán bao nhiêu | 0.417 | faithfulness |
| 6 | multi_hop | q25 — Lương thử việc Junior mức cao nhất | 0.500 | faithfulness |
| 7 | multi_hop | q37 — Tự ý xoá malware + chia sẻ sự cố lên Slack | 0.500 | faithfulness |
| 8 | adversarial | q50 — Manager dùng VPN cá nhân khi WFH | 0.667 | answer_relevancy |
| 9 | factual | q9 — Nam nhân viên nghỉ bao nhiêu ngày khi vợ sinh | 0.667 | faithfulness |
| 10 | multi_hop | q38 — Công tác nước ngoài 4 ngày, khách sạn 200 USD/đêm | 0.700 | faithfulness |

**7/10 câu tệ nhất có câu trả lời là "Không tìm thấy."** — pipeline từ chối trả lời
chứ không trả lời sai. Đây là một failure mode rất khác với hallucination, và cần
cách sửa khác hẳn.

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 2 | **13** | 1 | 16 |
| answer_relevancy | **14** | 3 | 2 | 19 |
| context_precision | 1 | 0 | 0 | 1 |
| context_recall | 3 | 4 | **7** | 14 |

Ma trận này cho thấy **ba cụm lỗi tách bạch**, không phải một lỗi chung lan ra:

- **multi_hop → faithfulness** (13/20): câu cần tổng hợp nhiều chunk và tính toán.
- **factual → answer_relevancy** (14/20): câu đơn giản, trả lời đúng nhưng cụt.
- **adversarial → context_recall** (7/10): câu cài bẫy xung đột phiên bản chính sách.

---

## 4. Dominant Failure Analysis

**Dominant distribution:** `multi_hop` (avg_score = 0.716, thấp nhất)
**Dominant metric:** `answer_relevancy` (worst_metric của 19/50 câu, và cũng là
metric có điểm trung bình thấp nhất: 0.693)

**Lý do phân tích:**

> `multi_hop` yếu nhất vì câu hỏi buộc pipeline làm hai việc mà RAG thuần không làm
> được: ghép dữ kiện nằm ở nhiều chunk khác nhau, rồi **tính toán** trên chúng
> (q31 "công tác 2 ngày × 1.5tr/đêm", q33 "Manager 12 năm → tổng phụ cấp + phép").
> Khi gặp bài toán nhiều bước, model chọn phương án an toàn là trả lời "Không tìm
> thấy." — đúng theo system prompt "nếu context không chứa thông tin thì trả lời
> Không tìm thấy". Kết quả: faithfulness = 0 (không có statement nào để verify) và
> answer_relevancy = 0 cùng lúc, kéo avg_score xuống 0.375-0.5.
>
> Bằng chứng quyết định: **q25 và q37 có context_precision = 1.00 VÀ context_recall
> = 1.00** mà vẫn trả lời "Không tìm thấy". Retrieval đã lấy về đúng chunk chứa câu
> trả lời, LLM vẫn từ chối. Vậy nút thắt nằm ở **generation**, không phải retrieval —
> tăng top-k hay đổi embedding model sẽ không cứu được nhóm này.
>
> `answer_relevancy` thấp nhất vì một cơ chế rất cụ thể của RAGAS: metric này gắn cờ
> *noncommittal* và cho thẳng **0 điểm** với câu trả lời né tránh. Cả 9 câu có
> answer_relevancy = 0 đều là loại này: 7 câu "Không tìm thấy.", q50 "Không.",
> q6 "Bí mật.". Đây **không phải** thiên vị câu ngắn — q8 chỉ dài 16 ký tự nhưng có
> nội dung thật vẫn đạt 0.93. Nói cách khác, 0.693 không phản ánh "câu trả lời lạc
> đề" mà phản ánh "pipeline im lặng quá nhiều".
>
> `adversarial` giữ được faithfulness cao (0.855) — pipeline **không** bịa khi bị gài
> bẫy, đó là tin tốt. Nhưng context_recall chỉ 0.717 (thấp nhất trong ba nhóm): bộ
> câu adversarial cố tình hỏi những điều tồn tại ở nhiều phiên bản chính sách, và
> retriever lấy về đúng một phiên bản thay vì đủ cả hai để so sánh.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness (0.753) | Câu multi-hop cần ghép ≥2 chunk + tính toán; model chọn từ chối thay vì suy luận. 7/9 ca faithfulness=0 nằm ở multi_hop. | Tách generation thành 2 bước: (1) trích dữ kiện liên quan từ từng chunk, (2) suy luận/tính toán trên các dữ kiện đã trích. Cho phép model trình bày phép tính. Nới `RERANK_TOP_K` từ 3 lên 5 riêng cho câu multi-hop. |
| answer_relevancy (0.693) | 9 câu bị RAGAS gắn cờ *noncommittal* → 0 điểm. Nguyên nhân gốc là system prompt ép "Không tìm thấy" quá dễ dãi. | Sửa system prompt: chỉ được trả "Không tìm thấy" khi context **thực sự** không có dữ kiện nào liên quan; nếu có một phần thì phải trả lời phần đó kèm nêu rõ phần còn thiếu. Với câu yes/no, bắt buộc kèm căn cứ ("Không — theo mục X, VPN cá nhân bị cấm") thay vì "Không." trống. |
| context_recall (0.823) | Adversarial hỏi xuyên nhiều phiên bản chính sách, retriever trả về một phiên bản. | Đưa `policy_version` vào metadata chunk (M5 đã sinh sẵn auto_metadata) và khi câu hỏi có dấu hiệu so sánh phiên bản thì buộc lấy đủ các version. |
| context_precision (0.963) | Không phải điểm yếu — hybrid search + cross-encoder rerank hoạt động tốt. | Giữ nguyên. Đây là phần **không** nên tối ưu thêm; công sức nên dồn vào generation. |

---

## 6. Nhận xét về Adversarial Distribution

> avg_score của `adversarial` là **0.808**, nằm giữa `factual` (0.901) và `multi_hop`
> (0.716) — tức là pipeline **không** bị bộ câu cài bẫy đánh gục như dự đoán ban đầu.
>
> Quan trọng hơn con số tổng: `faithfulness` của adversarial đạt **0.855**, cao hơn
> hẳn multi_hop (0.521). Nghĩa là khi bị hỏi bẫy về xung đột phiên bản (v2023 vs
> v2024), pipeline **không bịa ra câu trả lời** — nó bám context hoặc từ chối. Với
> một trợ lý tra cứu chính sách nội bộ thì đây đúng là hành vi mong muốn: trả lời sai
> một quy định lương thưởng nguy hiểm hơn nhiều so với việc nói "không tìm thấy".
>
> Điểm yếu thật của nhóm này là `context_recall` = **0.717**, thấp nhất cả ba nhóm.
> Bẫy phiên bản không làm pipeline nói dối, nhưng làm nó **nhìn thiếu**: retriever
> lấy về một phiên bản chính sách trong khi câu hỏi cần đối chiếu cả hai.
>
> Trong bottom 10 có 2 câu adversarial: **q44** (rank 2) và **q50** (rank 8). q44
> "Bao lâu phải đổi mật khẩu một lần?" có context_recall = 1.00 — chunk đúng đã được
> lấy về — nhưng pipeline vẫn trả "Không tìm thấy.", lặp lại đúng lỗi generation của
> nhóm multi_hop. q50 "Manager dùng VPN cá nhân khi WFH" thì trả lời **đúng** ("Không")
> với faithfulness = 1.00, chỉ mất điểm answer_relevancy vì câu trả lời trống rỗng
> không kèm căn cứ. Cả hai đều là lỗi trình bày/generation chứ không phải bị lừa.
