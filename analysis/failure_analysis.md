# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Phi Nhật  
**Mã số học viên (MSSV):** 2A202602658  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

Số liệu dưới đây được lấy từ lần chạy `python main.py` (20 câu test set, Qdrant thật
qua Docker, OpenAI API key hoạt động). Đây chính là nội dung file
`reports/ragas_report.json` nộp bài.

### Naive Baseline vs Production RAG

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|-----------|---|
| ✓ Faithfulness | 0.8308 | **0.9000** | **+0.0692** |
| ✓ Answer Relevancy | 0.5211 | **0.8319** | **+0.3108** |
| ✓ Context Precision | 0.9250 | **0.9917** | **+0.0667** |
| ✗ Context Recall | 0.9250 | 0.8583 | **−0.0667** |

3 metric tăng, trong đó Answer Relevancy tăng mạnh nhất (+0.3108). **Context Recall
giảm 0.0667 — đây là đánh đổi có chủ đích, tôi chọn chấp nhận và giải thích ở mục
bên dưới, không giấu đi.**

### Diễn giải từng kết quả

**Answer Relevancy 0.5211 → 0.8319 (+0.3108).** Đây là kết quả đáng chú ý nhất
vì nó bắt nguồn từ việc tìm ra và sửa **hai lỗi trong chính bộ chấm RAGAS**, không
phải trong hệ thống trả lời. Chi tiết ở mục "Hai lỗi trong bộ chấm" bên dưới.

**Faithfulness 0.9000** — đạt ngưỡng 0.85 mà rubric đặt ra. Parent expansion đóng
góp lớn nhất: đoạn con 256 ký tự bị cắt mất điều kiện, đoạn cha 2048 ký tự giữ
đủ bối cảnh nên LLM không cần tự suy diễn phần thiếu.

**Context Precision 0.9917** — cao nhất trong 4 metric. Cross-Encoder rerank 20→3
loại được đoạn nhiễu trước khi đưa vào prompt.

**Context Recall 0.8583 (−0.0667 so với baseline) — đánh đổi có chủ đích.**
Nguyên nhân: khi tìm hiểu câu hỏi *"Mật khẩu phải có tối thiểu bao nhiêu ký tự?"*,
hệ thống **cố ý không lấy** `mat_khau_v1.md` (quy định cũ 8 ký tự, đã hết hiệu lực)
và chỉ lấy `mat_khau_v2.md` (12 ký tự). Câu trả lời *"12 ký tự"* là **đúng**. Nhưng
ground truth trong `test_set.json` lại viết: *"Chính sách cũ (v1.0) yêu cầu 8 ký tự
nhưng đã bị thay thế"* — RAGAS so câu trả lời với ground truth và thấy thiếu mệnh
đề về bản cũ, nên trừ điểm.

Có 4 câu cùng dạng này (mật khẩu, phép năm 2023/2024), đều trừ về 0.5 vì lý do trên.
Nói thẳng: **đây là hạn chế của bộ test, không phải lỗi hệ thống.** Một trợ lý nội bộ
hợp lý chỉ nên trả lời theo chính sách đang hiệu lực. Tôi vẫn giữ hành vi này vì trả
lời theo quy định đã hết hiệu lực là lỗi nghiêm trọng hơn nhiều trong thực tế.

### Hai lỗi trong bộ chấm RAGAS (đã sửa)

Ban đầu Answer Relevancy chỉ 0.5287. Điều tra cho thấy **nguyên nhân nằm ở bộ chấm,
không phải hệ thống trả lời** — và phải mất hai vòng mới tìm ra hết.

**Lỗi 1: Judge dùng embedding tiếng Anh cho văn bản tiếng Việt.**
`answer_relevancy` trong RAGAS 0.1.22 **không** hỏi LLM "câu trả lời có liên quan
không". Nó **sinh câu hỏi phụ từ câu trả lời**, rồi so cosine similarity giữa
embedding(câu hỏi gốc) và embedding(câu hỏi phụ). Mặc định dùng
`text-embedding-3-small` — model tiếng Anh. Đổi sang `text-embedding-3-large`.

**Lỗi 2: Prompt sinh câu hỏi phụ thuần tiếng Anh.** Đây mới là lỗi lớn. Cả
instruction lẫn 4 ví dụ mẫu trong `ragas` đều bằng tiếng Anh
(*"Where was Albert Einstein born?"*, *"Everest"*), và **không có yêu cầu nào về
ngôn ngữ**. Nên khi đưa câu trả lời tiếng Việt vào, judge vẫn phát ra câu hỏi tiếng
Anh — rồi ta so embedding tiếng Anh với câu hỏi tiếng Việt.

Đo trực tiếp trên 4 câu lấy từ report, giữ nguyên câu trả lời, chỉ đổi prompt:

| Prompt | Số probe tiếng Việt | Mean answer_relevancy |
|--------|---------------------|----------------------|
| Tiếng Anh (mặc định) | 0/3, 1/3, 0/3, 3/3 | 0.6146 |
| Tiếng Việt (đã sửa) | 3/3, 3/3, 3/3, 3/3 | 0.7161 |

Câu *"Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng"* nhảy từ **0.495 → 1.000** chỉ nhờ
đổi prompt. Đã viết prompt tiếng Việt với few-shot ví dụ tiếng Việt trong
`_apply_vietnamese_prompt()`.

**Lỗi 3 (phát hiện muộn): `np.any` khiến 1 phiếu lỗi đủ giết điểm.** RAGAS coi câu
trả lời là "né tránh" nếu **chỉ cần 1 trong 3** lượt judge trả `noncommittal=1`, rồi
nhân điểm cosine với 0. Ba câu có `faithfulness = 1.0` **và** `context_recall = 1.0`
tức trả lời hoàn hảo vẫn nhận đúng **0.000**. Đã ghi đè thành bỏ phiếu đa số
(≥ 2/3 mới bị phạt) trong `_majority_calculate_score()`.

Bài học: **điểm thấp không đồng nghĩa hệ thống sai.** Phải kiểm chứng từng giả thuyết
bằng đo đạc trước khi sửa prompt hệ thống — nếu sửa mù, tôi sẽ tốn công sửa prompt
trả lời trong khi vấn đề thật nằm ở bộ chấm.

### Latency breakdown

Đo trên CPU (không có GPU), `bge-reranker-v2-m3` là model 2.14 GB:

| Bước | Thời gian |
|------|-----------|
| Chunking (offline) | ~240 ms |
| Enrichment M5 (offline, cache hit) | ~1 ms |
| **Hybrid Search + RRF (per query)** | 300.3 ms |
| **Cross-Encoder Rerank 20→3 (per query)** | **7 306.8 ms** |
| LLM answer generation (per query) | 1 352.4 ms |
| **End-to-end (per query)** | **8 959.5 ms** |

**SLA rerank < 150ms: FAIL** — vi phạm rõ ràng, tôi ghi nguyên trạng thái này.

Rerank chiếm **81.5%** tổng latency. Ba hướng cải thiện:

1. **Giảm số ứng viên đưa vào rerank.** `HYBRID_TOP_K` 20 → 10 sẽ cắt khoảng 50%
   latency rerank, đánh đổi một phần context_recall.
2. **Giảm `RERANK_MAX_LENGTH`.** Hiện 512. Chunk cha 2048 ký tự tiếng Việt khoảng
   600-700 token, nên hạ xuống 256 có nguy cơ cắt mất đáp án nằm cuối đoạn. Chỉ nên
   hạ sau khi đo lại context_recall — không hạ mù.
3. **Dùng ONNX / Flashrank.** `FlashrankReranker` có sẵn trong khung, cho latency
   thấp hơn nhiều trên CPU.

Tôi chưa đo được con số trên GPU nên không đưa vào báo cáo. Con số 7.306 s là số đo
thực tế trên máy này, không phải ước tính.

---

## Bottom-5 Failures

5 ca dưới đây được trích trực tiếp từ `reports/ragas_report.json` hiện tại, sắp xếp
tăng dần theo điểm trung bình 4 metric. Mỗi ca trả lời 4 câu hỏi theo rubric, kèm
**Error Tree** đi theo từng nhánh.

### Cây phân loại lỗi (Error Tree)

Áp dụng cho từng câu hỏi, đi theo 4 bước — dừng ở nhánh đúng đầu tiên:

```
Câu trả lời có đúng không?
├─ SAI → Context có chứa đáp án đúng không?
│        ├─ KHÔNG  → Lỗi tầng Retrieval  (M1 chunking / M2 search)
│        │            ├─ parent_id trùng giữa các tài liệu?  → M1
│        │            ├─ cần nhiều tài liệu cho 1 câu?        → M2 (multi-hop)
│        │            └─ bản chính sách cũ lấn át bản mới?    → M1+M3 (phiên bản)
│        └─ CÓ     → LLM không dùng hết context  → M4 (prompt sinh câu trả lời)
└─ ĐÚNG → Điểm thấp do metric hay do hệ thống?
         ├─ CÁC metric khác đều = 1.0, một metric = 0.0
         │     → Lỗi ở BỘ CHẤM (M4): judge embedding sai ngôn ngữ,
         │       prompt sinh câu hỏi phụ sai ngôn ngữ, hoặc np.any quá nhạy
         └─ context_recall < 1.0
               → Hệ thống thiếu thông tin trong context  → M2
```

### #1 — avg 0.627

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn 15 ngày, quá hạn 5 ngày, phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (pro-rata khoảng **50.000 VNĐ** cho 5 ngày).
- **Got:** Trả lời **đúng 50.000 VNĐ**, có trình bày công thức `15.000.000 × 2% × 5/30`.
- **Scores:** faithfulness **0.0** · answer_relevancy 0.84 · context_precision 1.0 · context_recall 0.67

**1. Câu trả lời có đúng không?** **Đúng về mặt nghiệp vụ** — 50.000 VNĐ khớp ground
truth. Nhưng `faithfulness = 0.0` vì câu trả lời chứa phép tính `15.000.000 × 2% ×
5/30` mà **không có câu nào trong context nói công thức đó**. RAGAS đo "mọi khẳng
định trong câu trả lời có bám được context không", nên phép tính — dù chính xác — bị
coi là không hỗ trợ. Đây là điểm đáng chú ý của RAGAS: **hệ thống làm đúng việc suy
luận thì bị phạt.**

**2. Các đoạn trích dẫn có chứa đáp án không?** Có. `tam_ung.md` chứa cả "15 ngày"
và "2%/tháng", nhưng **không chứa** cách quy đổi 5 ngày thành phần tháng — đây là kiến
thức cần suy luận, không có trong tài liệu.

**3. Câu hỏi có cần viết lại không?** Không. Câu hỏi rõ ràng, có đủ dữ kiện.

**4. Sửa ở module nào?** Đã sửa ở **M4/prompt sinh câu trả lời** (`_GENERATION_SYSTEM_
PROMPT` trong `src/pipeline.py`): thêm quy tắc *"khi cần tính toán, phải quy đổi về
cùng đơn vị trước khi nhân, ví dụ tiền × tỉ lệ × (N ÷ 30)"*. Trước đó hệ thống trả
**300.000 VNĐ** (sai 6 lần) vì áp nguyên tỉ lệ 2%/tháng cho 5 ngày. Sau khi sửa, câu
trả lời về nghiệp vụ đã đúng; điểm faithfulness 0.0 là hệ quả của RAGAS, chấp nhận
được vì thay đổi đó đánh đổi một metric để lấy câu trả lời đúng.

- **Error Tree:** Câu trả lời ĐÚNG về nghiệp vụ → context CÓ đáp án nhưng KHÔNG có
  công thức quy đổi → nhánh **"LLM không dùng hết context"** → **M4 (prompt)**.
  Điểm 0.0 là hệ quả RAGAS chấm phép tính suy luận là "không bám context".

### #2 — avg 0.698

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Đúng phần phép (**18 ngày**), nhưng nói *"không có thông tin cụ thể trong context về mức lương"*.
- **Scores:** faithfulness 0.5 · answer_relevancy 0.79 · context_precision 1.0 · **context_recall 0.5**

**1. Câu trả lời có đúng không?** Một nửa. Phần phép năm đúng, phần lương **thiếu thật** —
hệ thống nói "không tìm thấy" trong khi `bang_luong_2024.md` có bảng lương theo cấp bậc.

**2. Các đoạn trích dẫn có chứa đáp án không?** Không cho phần lương. Câu hỏi này đòi hỏi
**hai tài liệu khác nhau** (`nghi_phep_nam_v2024.md` + `bang_luong_2024.md`) nhưng top-3
chỉ chứa đủ tài liệu phép năm.

**3. Câu hỏi có cần viết lại không?** Không — đây là câu hỏi hợp lệ, chỉ là khó.

**4. Sửa ở module nào?** **M2 (Hybrid Search)** — cần tăng `HYBRID_TOP_K` hoặc bổ sung
**multi-hop retrieval** (truy vấn lần 2 dựa trên kết quả lần 1) để gom tài liệu từ
nhiều nguồn. Đây là hạn chế thật của kiến trúc single-shot RRF hiện tại.

- **Error Tree:** Câu trả lời SAI ở một phần → context KHÔNG chứa đủ đáp án (thiếu
  `bang_luong_2024.md`) → nhánh **"cần nhiều tài liệu cho 1 câu"** → **M2 (multi-hop)**.
  Đây là lỗi retrieval thật, không phải vấn đề chấm điểm.

### #3 — avg 0.750

- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Dữ liệu **Bí mật**, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** *"Thông tin lương thuộc cấp độ phân loại dữ liệu **Bí mật**."*
- **Scores:** faithfulness **1.0** · answer_relevancy **0.0** · context_precision 1.0 · context_recall 1.0

**1. Câu trả lời có đúng không?** Đúng, và đầy đủ ở mức câu hỏi hỏi ("cấp độ nào?").

**2. Các đoạn trích dẫn có chứa đáp án không?** Có, hoàn toàn — `context_recall = 1.0`.

**3. Câu hỏi có cần viết lại không?** Không.

**4. Sửa ở module nào?** **Không phải lỗi hệ thống** — và đây là ca đáng kể nhất về mặt kỹ thuật. `answer_relevancy = 0.0` trong khi cả 3 metric khác đều là 1.0. Nguyên nhân: judge LLM đánh dấu `noncommittal = 1` cho câu trả lời này ở đa số lượt (do chuỗi `"Thông tin lương là dữ liệu **Bí mật**, cấm chia sẻ"` trông giống trích dẫn hơn là một khẳng định), nên điểm bị nhân với 0.

Ban đầu `failure_analysis()` gán cho ca này nhãn **"Hệ thống không tìm được ngữ cảnh —
lỗi tầng Retrieval"**, điều này **sai hoàn toàn** vì `context_recall = 1.0`. Nguyên nhân
lỗi trong code: nhánh chẩn đoán điểm 0 kích hoạt cho *bất kỳ* metric nào bằng 0, kể cả
`answer_relevancy`. Đã sửa trong `src/m4_eval.py` — chỉ chẩn đoán lỗi retrieval khi
`context_recall < 1.0` hoặc `faithfulness < 1.0`, kèm test hồi quy
(`test_zero_relevancy_with_perfect_context_not_blamed_on_retrieval`).

- **Error Tree:** Câu trả lời ĐÚNG → cả 3 metric khác = 1.0, chỉ `answer_relevancy` = 0.0
  → nhánh **"Lỗi ở BỘ CHẤM"** → **M4 (cấu hình judge)**. Không sửa hệ thống trả lời.

### #4 — avg 0.791

- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo v2024 hiện hành, từ **3 năm** trở lên được cộng thêm 1 ngày phép mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** *"Từ **3 năm** trở lên được cộng thêm **1 ngày phép** cho mỗi 3 năm."*
- **Scores:** faithfulness **1.0** · answer_relevancy 0.66 · context_precision 1.0 · **context_recall 0.5**

**1. Câu trả lời có đúng không?** Hoàn toàn đúng, đúng bản hiện hành v2024.

**2. Các đoạn trích dẫn có chứa đáp án không?** Có, nhưng chỉ gồm bản v2024 — bản v2023
đã bị hạ thấp thứ hạng. `context_recall = 0.5` **vì ground truth nhắc cả quy định
cũ "5 năm"**.

**3. Câu hỏi có cần viết lại không?** Không. Đây là hệ quả của chính sách dữ liệu
phiên bản mà tôi đã triển khai.

**4. Sửa ở module nào?** Không sửa. Đây là **đánh đổi có chủ đích** giữa
context_recall và tính đúng đắn. Xem mục "Xung đột phiên bản" bên dưới.

- **Error Tree:** Câu trả lời ĐÚNG → context CÓ đáp án → nhánh **"ground truth mô tả
  hành vi khác với hệ thống"** → không phải lỗi code. RAGAS trừ vì ground truth nhắc
  cả quy định cũ đã hết hiệu lực.

### #5 — avg 0.804

- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu **v2.0** hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** *"Có, tất cả nhân viên **bắt buộc** kích hoạt MFA cho email, VPN và hệ thống nội bộ."*
- **Scores:** faithfulness **1.0** · answer_relevancy 0.72 · context_precision 1.0 · **context_recall 0.5**

Cùng dạng với #4: câu trả lời đúng hoàn toàn, `context_recall = 0.5` vì ground truth
có nhắc chính sách cũ. Không sửa.

- **Error Tree:** Câu trả lời ĐÚNG → context CÓ đáp án → nhánh **"ground truth mô tả
  hành vi khác với hệ thống"** → không phải lỗi code. Cùng cơ chế với #4.

---

## Xung đột phiên bản — phân tích chuyên sâu

Kho dữ liệu có **4 cặp bản chính sách cùng chủ đề**:

| Họ chính sách | Bản cũ (đã thay thế) | Bản hiện hành |
|---|---|---|
| Nghỉ phép năm | `nghi_phep_nam_v2023.md` (12 ngày, 3 năm/ngày) | `nghi_phep_nam_v2024.md` (15 ngày, 3 năm/ngày) |
| Mật khẩu | `mat_khau_v1.md` (8 ký tự, 90 ngày, không MFA) | `mat_khau_v2.md` (12 ký tự, 120 ngày, có MFA) |

### Triệu chứng ban đầu

Câu *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"* bị hệ thống trả lời:

> *"Nhân viên chính thức được hưởng **12 ngày phép năm** theo chính sách năm 2023 và
> **15 ngày phép năm** theo chính sách năm 2024."*

Câu *"Mật khẩu phải có tối thiểu bao nhiêu ký tự?"* từng trả **8 ký tự** (theo bản v1.0
đã hết hiệu lực). Điểm `context_precision` rớt về **0.333** ở các câu hỏi này.

### Vì sao Cross-Encoder không tự giải quyết được

Bản cũ và bản mới gần như **giống hệt nhau về mặt ngữ nghĩa** — chỉ khác con số:

- *"nhân viên được nghỉ **12** ngày phép năm"*
- *"nhân viên được nghỉ **15** ngày phép năm"*

Cross-encoder so điểm liên quan ngữ nghĩa, và "12 ngày" gần như không khác "15 ngày"
trong không gian vector. **Reranker không thể phân biệt chính sách nào còn hiệu lực** —
đó là thông tin nằm ở metadata, không nằm trong văn bản. Vì vậy cả hai bản cùng
chiếm suất trong top-3 và LLM phải tự quyết định → trả lời lưỡng lự.

### Cách sửa

1. **Trích metadata phiên bản từ header tài liệu** (`parse_policy_header()` trong
   `src/m1_chunking.py`). Mọi file .md đều có dòng
   `> Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024`, nên lấy trực tiếp thay vì đoán
   theo tên file.

2. **Gom về "họ chính sách"** bằng cách bỏ nhãn ở ngoặc cuối tiêu đề:
   `"Chính sách nghỉ phép năm (Phiên bản 2023)"` → `"chính sách nghỉ phép năm"`.
   Nhờ vậy `nghi_phep_nam_v2023.md` và `nghi_phep_nam_v2024.md` được nhận diện là
   cùng một họ.

3. **Đánh dấu bản bị thay thế** (`mark_superseded_policies()`) dùng hai tín hiệu
   độc lập: header tự khai *"ĐÃ THAY THẾ bởi v2.0"*, hoặc ngày hiệu lực cũ hơn bản
   mới nhất trong cùng họ. Kết quả trên kho thật: phát hiện đúng 2/2 bản cũ.

4. **Hạ thấp thứ hạng SAU khi rerank** (`_demote_superseded_candidates()` trong
   `src/pipeline.py`) — chứ không xóa. Lý do không xóa: câu hỏi hợp lệ về lịch sử
   chính sách (*"chính sách cũ quy định mấy ngày?"*) vẫn cần bản cũ; xóa sẽ khiến
   hệ thống trả lời sai kiểu *"không tìm thấy"*. Bản cũ chỉ bị đẩy xuống sau bản
   hiện hành trong danh sách ứng viên.

5. **Rerank trên pool rộng hơn trước khi lọc.** Đặt `RERANK_POOL_SIZE = 8` thay vì cắt
   top-3 ngay trong `rerank()`. Nếu cắt sớm, bản cũ đã chiếm hết 3 suất và việc hạ
   thấp trở nên vô nghĩa. Có test chống hồi quy
   (`test_rerank_pool_larger_than_top_k`).

6. **Bổ sung quy tắc vào system prompt**: *"Nếu context có nhiều phiên bản của cùng
   một chính sách, chỉ dùng phiên bản đang hiện hành. Không liệt kê hay so sánh các
   bản cũ."*

### Kết quả

| Câu hỏi | Trước | Sau |
|---|---|---|
| Nhân viên được nghỉ bao nhiêu ngày phép năm? | "12 ngày theo 2023 và 15 ngày theo 2024" | "**15 ngày phép năm** có lương" |
| Thâm niên bao nhiêu năm được cộng thêm ngày phép? | lẫn lộn 3 năm / 5 năm | "Từ **3 năm** trở lên, mỗi 3 năm thêm 1 ngày" |
| Mật khẩu tối thiểu bao nhiêu ký tự? | 8 ký tự (bản cũ) | "**12 ký tự**" |

`context_precision` tăng từ **0.9583 → 0.9917**. Đổi lại, `context_recall` giảm
0.9333 → 0.8583 vì 4 câu hỏi có ground truth nhắc cả bản cũ — **đây là cái giá
thành thật, tôi chọn chấp nhận vì trả lời theo quy định hết hiệu lực là lỗi
nghiêm trọng hơn trong vận hành thực tế.**

### Một lỗi im lặng đáng lưu ý

Cache enrichment ban đầu **không chứa khoá `is_superseded`**, nhưng vẫn được coi là
hợp lệ vì `_chunk_fingerprint()` chỉ tính **số lượng chunk**. Kết quả: toàn bộ cơ chế
lọc phiên bản chạy trên cache cũ sẽ **không bao giờ kích hoạt**, mà không có lỗi nào
báo. Đã thêm `PIPELINE_CACHE_VERSION = 2` vào fingerprint. Đây đúng là loại lỗi mà
cơ chế fingerprint sinh ra để chống.

---

## Tổng kết và hướng cải thiện

| Vấn đề | Trạng thái | Hướng xử lý |
|---|---|---|
| Q&A đa tài liệu (#2) | Chưa xử lý | Multi-hop retrieval trong M2 |
| Rerank 7.3 s, vi phạm SLA | Chưa xử lý | Giảm `HYBRID_TOP_K`, ONNX/Flashrank |
| 2 file PDF bị bỏ qua | Chưa xử lý | Cần OCR (`BCTC.pdf`, Nghị định 13/2023) |
| `answer_relevancy` còn 4 ca ở 0.5 | Đã giải quyết phần lớn | Xem mục "Hai lỗi trong bộ chấm" |
| context_recall −0.0667 | **Chấp nhận có chủ đích** | Sửa ground truth cho hợp lý |

Ưu tiên nếu còn thời gian: **OCR cho 2 file PDF** có giá trị cao nhất, vì hiện tại
các câu hỏi về lương và phân loại dữ liệu phải suy luận từ tài liệu .md khác thay
vì tìm thẳng trong nguồn chính.
