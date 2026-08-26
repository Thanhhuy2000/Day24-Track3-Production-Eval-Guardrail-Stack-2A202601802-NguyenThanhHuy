# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Thanh Huy (2A202601802)
**Ngày:** 26/08/2026
**Judge model:** `gemma-4-31b-it`
**Generator được chấm:** `gemini-3.5-flash-lite`

> Ghi chú chọn judge: judge thuộc **họ model khác** generator để tránh
> self-preference bias (model có xu hướng chấm cao câu trả lời do chính nó sinh ra).

---

## 1. Pairwise Judge Results

Chạy `pairwise_judge()` + `swap_and_average()` trên 10 cặp.
**A = câu trả lời của pipeline, B = ground truth.**

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| q1 | Nghỉ bao nhiêu ngày khi kết hôn | B | "Cả hai đều **chính xác**. Tuy nhiên B đầy đủ hơn khi làm rõ không trừ phép năm" |
| q5 | Mua thiết bị 55tr cần ai duyệt | tie | B cung cấp căn cứ ngưỡng 50 triệu |
| q12 | Thưởng Tết tối thiểu | tie | A trực tiếp súc tích; B chi tiết hơn |
| q21 | Senior 9 năm nghỉ bao nhiêu ngày | B | "Cả hai đều **chính xác và đầy đủ**. Tuy nhiên B cung cấp căn cứ cụ thể" |
| q23 | Tài trợ khoá học 25tr, nghỉ sau 8 tháng | B | "Cả hai đều **chính xác** về kết quả. Tuy nhiên B có đầy đủ căn cứ" |
| q29 | Tạm ứng 8tr quá 30 ngày | B | B chi tiết hơn về phân cấp phê duyệt |
| q33 | Manager 12 năm: phụ cấp + phép | B | "Cả hai đều **chính xác và đầy đủ**. Tuy nhiên B chi tiết hơn" |
| q41 | Nghỉ bao nhiêu ngày phép năm | tie | — |
| q46 | Thử việc có được nghỉ phép năm | **A** | "A chính xác, đầy đủ và súc tích. B chứa thông tin thừa" |
| q50 | Manager dùng VPN cá nhân khi WFH | B | B dẫn chiếu cụ thể phiên bản chính sách v1.3 |

---

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| q1 | B | B | B | ✓ |
| q5 | B | tie | tie | ✗ |
| q12 | A | B | tie | ✗ |
| q21 | B | B | B | ✓ |
| q23 | B | B | B | ✓ |
| q29 | B | B | B | ✓ |
| q33 | B | B | B | ✓ |
| q41 | tie | tie | tie | ✓ |
| q46 | A | A | A | ✓ |
| q50 | B | B | B | ✓ |

**Position bias rate: 20%** (2/10 — q5 và q12 đổi kết quả khi hoán vị vị trí)

Swap-and-average đã làm đúng việc: cả 2 ca không nhất quán đều bị hạ xuống `tie`
thay vì lấy bừa kết quả của một lượt. Nếu chỉ chạy `pairwise_judge()` một lượt, q12
sẽ được ghi là "A thắng" hoàn toàn do may rủi thứ tự trình bày.

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 6 label=1, 4 label=0)
**Judge labels:** quy ước của scaffold — judge chọn B (ground truth) → pipeline kém → 0;
judge chọn A hoặc tie → 1.

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| q1 | 1 | 0 | ✗ |
| q5 | 0 | 1 | ✗ |
| q12 | 1 | 1 | ✓ |
| q21 | 1 | 0 | ✗ |
| q23 | 1 | 0 | ✗ |
| q29 | 0 | 0 | ✓ |
| q33 | 1 | 0 | ✗ |
| q41 | 0 | 1 | ✗ |
| q46 | 1 | 1 | ✓ |
| q50 | 0 | 0 | ✓ |

**Cohen's κ = −0.1538**
**Interpretation:** poor — tệ hơn cả đoán ngẫu nhiên. Agreement rate chỉ 4/10.

### 3b. Vì sao κ âm: lỗi lệch construct, không phải judge kém

Đọc kỹ reasoning của judge sẽ thấy κ âm **không** phải vì judge đánh giá dở, mà vì
judge và human đang trả lời **hai câu hỏi khác nhau**:

- **Human** trả lời: *"câu trả lời của pipeline có ĐÚNG không?"*
  (q41 "12 ngày phép" → sai → 0; q50 "được dùng VPN" → sai → 0)
- **Judge** trả lời: *"A hay B tốt hơn?"* với B = ground truth — mà ground truth
  theo định nghĩa luôn là bản đầy đủ nhất.

Ở 4 câu **q1, q21, q23, q33**, judge nói nguyên văn *"Cả hai câu trả lời đều chính
xác"* rồi vẫn chọn B vì "đầy đủ hơn". Tức là judge **xác nhận A đúng**, nhưng quy ước
map winner→label lại ghi thành 0 (pipeline kém). Cả 4 câu này human đều gán 1.
Chính 4 câu này tạo ra toàn bộ phần κ âm.

Kiểm chứng định lượng — giữ nguyên judge, chỉ sửa cách map ở 4 câu mà judge tự nói
"cả hai đều chính xác" thành label 1:

| Cách map | Agreement | Cohen's κ | Diễn giải |
|---|---|---|---|
| Theo scaffold (winner B → 0) | 4/10 | **−0.1538** | poor |
| Sửa construct ("A đúng là đủ" → 1) | 8/10 | **+0.5455** | moderate |

Cùng một judge, cùng một output, κ nhảy từ −0.15 lên +0.55. Kết luận: **thiết kế của
phép đo sai, không phải judge sai.** Muốn đo "pipeline trả lời có đúng không" thì phải
dùng **reference-based grading** (chấm A theo rubric đúng/sai, đối chiếu ground truth),
chứ không phải pairwise preference.

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie) — 7 case:

- A thắng + A dài hơn B: **0** / 7 cases
- B thắng + B dài hơn A: **6** / 7 cases
- **Verbosity bias rate: 86%**

**Bối cảnh quan trọng:** ground truth (B) **dài hơn pipeline answer (A) ở cả 10/10 câu**
(ví dụ q50: 214 vs 38 ký tự). Khi một phía luôn dài hơn, chỉ số verbosity bias 86%
bị **lẫn với** yếu tố "B cũng đầy đủ hơn thật". Không thể kết luận dứt khoát judge
thiên vị độ dài chỉ từ con số này.

Nhưng có một ca tách bạch được hai yếu tố: **q46**, judge chọn **A dù A ngắn hơn**
(81 vs 175 ký tự), với lý do "B chứa thông tin thừa không liên quan". Nghĩa là judge
này **có** khả năng phạt phần thừa — nó không mù quáng chọn câu dài.

**Kết luận:** thiên lệch quan sát được nghiêng về **completeness bias** (ưu tiên câu
có dẫn chiếu, có căn cứ) hơn là verbosity bias thuần tuý. Điều này vẫn là vấn đề trong
production: người dùng tra cứu chính sách thường muốn câu trả lời ngắn gọn đúng trọng
tâm, nếu judge luôn ưu ái bản dài thì việc dùng nó để tinh chỉnh prompt sẽ đẩy hệ thống
sinh ra câu trả lời ngày càng dài dòng.

---

## 5. Nhận xét chung

> **κ = −0.154 chưa đạt ngưỡng tin cậy** (cần > 0.6), nhưng nguyên nhân đã truy được
> tới gốc và không nằm ở năng lực của judge: pairwise preference so với ground truth
> đo "câu nào hoàn chỉnh hơn", trong khi human label đo "câu của pipeline có đúng
> không". Sửa đúng construct thì cùng judge đó cho κ = +0.55 (moderate). Bài học rút
> ra: **trước khi kết luận LLM judge không đáng tin, phải kiểm tra xem nó có đang được
> hỏi đúng câu hỏi mà mình muốn đo hay không.**
>
> **Position bias 20% là chấp nhận được** nhưng không thể bỏ qua. 2/10 câu đảo kết quả
> khi hoán vị vị trí — nếu chỉ chạy một lượt thì 20% kết quả là ngẫu nhiên. Swap-and-
> average đã hạ đúng 2 ca đó xuống `tie` thay vì ghi nhận kết quả may rủi.
> Chi phí: gấp đôi số API call. Với 10 câu thì rẻ, với 50k câu/ngày trong production
> thì cần cân nhắc — chỉ swap khi lượt đầu cho kết quả sát nhau.
>
> **Trong môi trường production**, tôi sẽ dùng judge như sau: (1) **không** dùng
> pairwise-vs-ground-truth để gác cổng chất lượng — thay bằng reference-based grading
> với rubric nhị phân đúng/sai; (2) luôn bật swap-and-average cho mọi so sánh A/B thật
> (ví dụ so hai phiên bản prompt), coi ca không nhất quán là `tie` chứ không lấy một
> lượt; (3) thêm ràng buộc "câu ngắn mà đủ ý không bị trừ điểm" vào prompt judge để
> chặn completeness bias đẩy hệ thống sang trả lời dài dòng; (4) giữ judge **khác họ
> model** với generator, và định kỳ hiệu chuẩn lại bằng một bộ nhãn người — κ phải
> được đo lại mỗi khi đổi model, chứ không đo một lần rồi tin mãi.

---

## Phụ lục: bug đã sửa trong quá trình đo

Lần chạy đầu, 4/10 câu bị ghi nhầm thành `tie` do `_parse_judge_json()` hỏng: fallback
regex `\{.*\}` là greedy, khi judge trả về hai object JSON liền nhau thì nó gộp cả hai
thành chuỗi không parse được. Đã thay bằng `json.JSONDecoder().raw_decode()` quét từ
từng vị trí `{`. Số liệu trong báo cáo này là của lần chạy sau khi sửa, với **0 lỗi
parse**. Nếu không sửa, κ sẽ bị tính trên dữ liệu rác mà bề ngoài vẫn trông bình thường.
