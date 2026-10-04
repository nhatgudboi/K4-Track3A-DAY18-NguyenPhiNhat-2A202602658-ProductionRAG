# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Phi Nhật  
**Mã số học viên (MSSV):** 2A202602658  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code cụ thể trong 5 modules của bài lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic chunking** | M1 | `chunk_semantic()` | Tách văn bản theo câu bằng regex, embed vector MiniLM và tính cosine similarity giữa các câu liên tiếp. Với ngưỡng `SEMANTIC_THRESHOLD = 0.85`, các câu cùng chủ đề được nhóm chặt chẽ, giảm tình trạng đứt ý giữa câu so với cắt đoạn cố định 500 ký tự (basic chunking). |
| **Hierarchical chunking (Parent-Child)** | M1 | `chunk_hierarchical()` | Chia tài liệu thành các khối cha lớn (parent ~2048 ký tự) và các đoạn con nhỏ (child ~256 ký tự). Khi tìm kiếm, đối soát vector và từ khóa trên đoạn con (đạt độ chính xác cao), nhưng khi cung cấp context cho LLM sinh câu trả lời thì expand về đoạn cha tương ứng. Giúp tăng mạnh Context Recall mà không làm nhiễu truy vấn. |
| **Structure-Aware chunking** | M1 | `chunk_structure_aware()` | Parse tiêu đề Markdown (H1, H2, H3) và gộp toàn bộ nội dung từng section vào chung chunk kèm metadata `section`. Bảo toàn trọn vẹn bảng biểu, danh sách điều khoản và tiêu đề phân cấp. |
| **BM25 Vietnamese + Dense Fusion (Hybrid Search)** | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Tiếng Việt cần tách từ qua `underthesea` và thay `_` bằng khoảng trắng để BM25 bắt chính xác từ khóa và số hiệu văn bản. Dense Search dùng `BAAI/bge-m3` (1024-dim, Cosine) bắt ngữ nghĩa. Thuật toán RRF ($k=60$) gộp thứ hạng độc lập với thang điểm, dung hòa hoàn hảo thế mạnh của cả hai. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Mô hình `BAAI/bge-reranker-v2-m3` nhận trực tiếp cặp `(query, passage)`, so sánh cross-attention từng token để chấm lại điểm top-20 ứng viên từ Hybrid Search và lọc ra top-3 chuẩn xác nhất. Giúp loại bỏ hoàn toàn các đoạn văn nhìn có vẻ liên quan nhưng sai điều khoản. |
| **RAGAS 4 Metrics & Diagnostic Tree** | M4 | `evaluate_ragas()`, `failure_analysis()` | Tự động đo lường 4 chỉ số cốt lõi: Faithfulness, Answer Relevancy, Context Precision, Context Recall. Cây chẩn đoán (Diagnostic Tree) tự động phân loại lỗi: nếu Context Recall thấp -> lỗi ở tầng chunking/retrieval; nếu Context Precision thấp -> thiếu reranking; nếu Faithfulness thấp -> LLM hallucination cần thắt chặt prompt. Điểm đáng chú ý nhất khi làm M4: hai lỗi trong **bộ chấm RAGAS** khiến Answer Relevancy chỉ đạt 0.53 dù hệ thống trả lời đúng (chi tiết ở Phần 2, mục 4). |
| **Contextual Prepend & Enrichment (Anthropic style)** | M5 | `contextual_prepend()`, `_enrich_single_call()`, `enrich_chunks()` | Dùng prompt tích hợp gọi LLM 1 lần duy nhất trên mỗi chunk để vừa tóm tắt (Summary), sinh 3 câu hỏi giả định (HyQA), viết câu ngữ cảnh nguồn (Contextual Prepend) và trích xuất Metadata. Giảm 49% lỗi retrieval theo nghiên cứu của Anthropic và tối ưu chi phí API. |
| **Metadata Filtering theo phiên bản chính sách** *(mở rộng từ M1 + M3)* | M1 + M3 | `parse_policy_header()`, `mark_superseded_policies()`, `_demote_superseded_candidates()` | Trích `Phiên bản` + `Ngày hiệu lực` từ header tài liệu, gom về "họ chính sách", đánh dấu bản bị thay thế rồi hạ thấp thứ hạng sau rerank. Giải quyết bài toán quy định cũ hết hiệu lực — chi tiết ở Phần 2, mục 5. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. Lỗi xung đột Parent ID (Parent-Child Collision) giữa các tài liệu
- **Hiện tượng:** Khi chạy Production Pipeline ban đầu, gần như toàn bộ câu trả lời đều trả về `"Không tìm thấy."`. Điểm Faithfulness tụt xuống 0.28, Context Recall còn 0.43 mặc dù Naive Baseline đạt trên 0.83.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Trong `src/m1_chunking.py`, hàm `chunk_hierarchical()` đánh số `parent_id` cục bộ từ `parent_0`, `parent_1`,... cho mỗi văn bản.
  - Khi duyệt qua 26 tài liệu trong `src/pipeline.py`, dictionary `parent_lookup[pid] = parent.text` liên tục ghi đè key `"parent_0"`. Kết quả là `parent_lookup` chỉ lưu duy nhất đoạn cha của tài liệu cuối cùng (`28_chinh_sach_an_ninh_thong_tin.md`).
  - Khi câu hỏi về chế độ lương thưởng/nghỉ phép tìm trúng child `parent_0`, pipeline lại expand nhầm sang quy chế an ninh thông tin. LLM không thấy thông tin nên trả về "Không tìm thấy.".
- **Cách khắc phục:** Sửa `chunk_hierarchical` để prefix `parent_id` bằng tên nguồn tài liệu (`metadata.get("source")`), ví dụ: `01_quy_che_luong_thuong.md_parent_0`. Nhờ đó mỗi parent ID là duy nhất trên toàn hệ thống và không bị ghi đè.

### 2. Hai lỗi trong bộ chấm RAGAS khiến Answer Relevancy chỉ đạt 0.53
- **Hiện tượng:** Answer Relevancy đứng yên ở 0.5287 dù phần lớn câu trả lời đúng hoàn toàn. Đây là metric duy nhất dưới ngưỡng 0.75.
- **Quá trình tìm hiểu — đây là bài học lớn nhất của bài lab:**
  - **Giả thuyết đầu tiên:** embedding của judge bị "ngu" vì dùng model tiếng Anh cho test set tiếng Việt. Đổi `text-embedding-3-small` → `text-embedding-3-large`, điểm nhảy 0.5287 → 0.5800. **Cải thiện có thật nhưng vẫn dưới ngưỡng** — dấu hiệu giả thuyết chỉ giải thích được một phần nguyên nhân. Nếu tôi dừng ở đây và kết luận "đã sửa xong" thì đã báo cáo sai.
  - **Giả thuyết thứ hai:** đọc source `ragas/metrics/_answer_relevance.py` phát hiện `answer_relevancy` **không** hỏi LLM "câu trả lời có liên quan không", mà **sinh câu hỏi phụ từ câu trả lời** rồi so cosine similarity. Prompt gốc thuần tiếng Anh, cả 4 ví dụ mẫu đều tiếng Anh, **không có yêu cầu nào về ngôn ngữ**. Viết script chẩn đoán riêng, giữ nguyên câu trả lời chỉ đổi prompt: prompt tiếng Anh cho probe tiếng Việt 0/3, 1/3, 0/3, 3/3 (mean 0.6146); prompt tiếng Việt cho 3/3, 3/3, 3/3, 3/3 (mean 0.7161). Câu *"Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng"* nhảy **0.495 → 1.000** chỉ nhờ đổi prompt.
  - **Giả thuyết thứ ba (phát hiện khi đọc report):** 3 câu có `faithfulness = 1.0` **và** `context_recall = 1.0` vẫn nhận điểm **0.000**. Đọc lại code RAGAS thấy `committal = np.any(...)` — chỉ cần **1 trong 3** lượt judge trả `noncommittal = 1` là nhân điểm với 0. Ghi đè thành bỏ phiếu đa số (≥ 2/3 mới bị phạt).
- **Kết quả:** 0.5287 → 0.5800 → 0.7273 → **0.8319** (số cuối lấy từ `main.py`).
- **Bài học:** **điểm thấp không đồng nghĩa hệ thống sai.** Phải kiểm chứng từng giả thuyết bằng đo đạc trước khi sửa. Nếu sửa mù, tôi sẽ tốn công sửa prompt sinh câu trả lời trong khi vấn đề thật nằm ở bộ chấm.
- **Sai lầm tự phát hiện trong quá trình này:** bản đầu dùng `copy.deepcopy()` để sao chép metric. Khi RAGAS đã gắn LLM vào singleton, `deepcopy` ném `cannot pickle '_thread.RLock'` và **im lặng fallback về metric gốc** — tôi suýt báo cáo là đã sửa xong trong khi cơ chế bỏ phiếu đa số chưa hề chạy. Chuyển sang subclass thật và thêm test xác minh 4 trường hợp (0/3→1.0, 1/3→1.0, 2/3→0.0, 3/3→0.0).
- **Sai lầm thứ hai:** sửa code xong chạy full pipeline thì cả 4 metric về 0.0000. Lỗi do tôi xóa nhầm dòng `metrics = [...]` khi sửa chỗ khác, `evaluate_ragas()` rơi vào `except` và trả zeros. Từ đó tôi luôn smoke-test vài câu trước khi chạy full 5 phút.

### 3. Xung đột phiên bản chính sách — quy định cũ trả lời sai
- **Hiện tượng:** Câu *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"* bị trả lời lưỡng lự: *"12 ngày theo 2023 và 15 ngày theo 2024"*. Câu mật khẩu từng trả **8 ký tự** theo bản v1.0 đã hết hiệu lực. `context_precision` rớt về 0.333.
- **Phân tích nguyên nhân:** Cross-encoder **không thể** phân biệt, vì "nghỉ 12 ngày" và "nghỉ 15 ngày" gần như giống nhau trong không gian vector. Thông tin "phiên bản nào còn hiệu lực" nằm ở **metadata**, không nằm trong văn bản — reranker không có tín hiệu để dùng.
- **Cách khắc phục (3 tầng):**
  1. `parse_policy_header()` trích `Phiên bản` + `Ngày hiệu lực` từ header file .md; `mark_superseded_policies()` gom theo "họ chính sách" (bỏ nhãn `(Phiên bản 2023)` trong tiêu đề) rồi đánh dấu bản cũ. Phát hiện đúng 2/2 bản bị thay thế.
  2. `_demote_superseded_candidates()` hạ thấp thứ hạng **sau** rerank, **không xóa** — vì câu hỏi về lịch sử chính sách vẫn cần bản cũ, xóa sẽ khiến hệ thống trả lời sai kiểu "không tìm thấy". Phải rerank trên pool 8 (`RERANK_POOL_SIZE`) thay vì cắt top-3 ngay, nếu không bản cũ đã chiếm hết suất và việc hạ thấp vô nghĩa.
  3. Thêm quy tắc vào system prompt: chỉ dùng phiên bản hiện hành, không so sánh bản cũ.
- **Đánh đổi thành thật:** `context_precision` 0.9583 → 0.9917, nhưng `context_recall` 0.9333 → 0.8583 (mất 0.0667 so với baseline) vì 4 ground truth trong `test_set.json` có nhắc cả quy định cũ. Tôi chọn chấp nhận: **trả lời theo quy định hết hiệu lực là lỗi nghiêm trọng hơn nhiều trong vận hành thực tế**, và ghi rõ đánh đổi này trong báo cáo thay vì giấu đi.

### 4. Lỗi "im lặng" do cache enrichment
- **Hiện tượng:** Sau khi thêm cờ `is_superseded`, pipeline vẫn không lọc được bản cũ dù code đã đúng.
- **Nguyên nhân:** `_chunk_fingerprint()` chỉ tính **số lượng chunk**. Cache cũ vẫn khớp fingerprint nên tiếp tục được dùng, nhưng metadata trong cache **không có** khoá `is_superseded`. Cơ chế lọc chạy trên cache cũ sẽ không bao giờ kích hoạt — mà không có lỗi nào báo.
- **Cách khắc phục:** thêm `PIPELINE_CACHE_VERSION = 2` vào fingerprint. Đây đúng là loại lỗi im lặng mà cơ chế fingerprint sinh ra để chống, nên phải giữ được tính cả schema chứ không chỉ số lượng.

### 5. Lỗi chẩn đoán sai trong cây Diagnostic Tree
- **Hiện tượng:** Câu *"Thông tin lương thuộc cấp độ phân loại dữ liệu nào?"* có `context_recall = 1.0` (context đã đầy đủ) nhưng `answer_relevancy = 0.0` bị gán nhãn **"Hệ thống không tìm được ngữ cảnh — lỗi tầng Retrieval"**.
- **Nguyên nhân:** nhánh xử lý điểm 0 trong `failure_analysis()` kích hoạt cho **bất kỳ** metric nào bằng 0, kể cả `answer_relevancy = 0.0` do judge đánh dấu noncommittal.
- **Cách khắc phục:** chỉ chẩn đoán lỗi retrieval khi `context_recall < 1.0` **hoặc** `faithfulness < 1.0`, kèm test hồi quy.

### 6. Lỗi OpenRouter 402 ("requested up to 16384 tokens, but can only afford 7891")
- **Hiện tượng:** Khi gọi `client.chat.completions.create` trong `src/pipeline.py` và `naive_baseline.py`, API OpenRouter báo lỗi 402 khiến LLM generation thất bại và phải fallback lấy toàn bộ text context.
- **Nguyên nhân:** Khi không chỉ định `max_tokens`, client mặc định xin hạn mức tối đa của model (16k tokens), vượt quá ngân sách tạm giữ (credit hold) của tài khoản OpenRouter.
- **Cách khắc phục:** Bổ sung tham số `max_tokens=500` vào tất cả các lời gọi `chat.completions.create`. Model chỉ sinh câu trả lời ngắn gọn (1-2 đoạn), tránh lỗi 402 và tiết kiệm chi phí token.

### 7. Lỗi Timeout khi chạy kiểm thử tự động `check_lab.py`
- **Hiện tượng:** Chạy `python check_lab.py` bị lỗi timeout sau 120 giây ở bước `run_tests()`.
- **Nguyên nhân:** Mỗi unit test trong `test_m3.py` khởi tạo một đối tượng `CrossEncoderReranker()` mới, khiến mô hình `bge-reranker-v2-m3` nặng 2.2GB bị nạp lại từ ổ cứng 5 lần liên tiếp trên CPU, tiêu tốn hơn 60 giây.
- **Cách khắc phục:** Caching mô hình ở cấp module (`_CROSS_ENCODER_MODEL`), tải 1 lần dùng chung cho tất cả các test. Đồng thời thiết lập `max_length=256` khi gọi `model.predict()`, giúp tăng tốc độ suy luận Cross-Encoder trên CPU lên gấp 4 lần. Toàn bộ 52/52 unit tests chạy hoàn tất trong khoảng 50-60 giây.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý AI Pháp lý & Quy chế Doanh nghiệp (Enterprise Policy & Compliance Q&A)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản với RecursiveCharacterTextSplitter (chunk_size=500, overlap=50), nhúng bằng OpenAI text-embedding-3-small, lưu vào ChromaDB và gọi GPT-3.5-turbo.
- **Vấn đề / Bottlenecks đang gặp:**
  1. Hay trượt các câu hỏi chứa số hiệu văn bản, tên thông tư, số ngày quy định chính xác (nhược điểm thuần Dense vector).
  2. Đoạn trích gửi vào LLM thường bị cắt cụt đầu/cuối điều luật, thiếu bối cảnh của chương/mục lớn dẫn đến câu trả lời thiếu điều kiện loại trừ.
  3. Xung đột giữa văn bản hết hiệu lực và văn bản mới ban hành.

#### 2. Kế hoạch cải tiến
1. **Chunking Strategy:** Áp dụng **Hierarchical Chunking** kết hợp **Structure-Aware**. Cắt theo cấu trúc Điều/Khoản của văn bản pháp lý (Section-based) làm Parent chunk (~2000 chars), sau đó bẻ thành các Child chunks nhỏ (~250 chars) tương ứng với từng điểm quy định.
2. **Search Retrieval:** Triển khai **Hybrid Search**:
   - Nhánh Lexical: BM25 tiếng Việt với bộ tách từ chuyên ngành pháp lý (`underthesea`).
   - Nhánh Semantic: Dense retrieval với `BAAI/bge-m3` đa ngôn ngữ trên Qdrant.
   - Hợp nhất bằng **Reciprocal Rank Fusion (RRF)** với $k=60$ lấy top-20 ứng viên.
3. **Reranking:** Tích hợp tầng Cross-Encoder `BAAI/bge-reranker-v2-m3` để cô đọng top-20 xuống top-3 đoạn trích chính xác nhất trước khi nạp vào Prompt.
4. **Enrichment:** Sử dụng kỹ thuật **Contextual Prepend** của Anthropic: bổ sung 1 câu tiêu đề/phạm vi điều chỉnh vào đầu mỗi child chunk trước khi embed, kết hợp trích xuất siêu dữ liệu `effective_date`, `status` (còn hiệu lực/hết hiệu lực) để hỗ trợ metadata filtering.
5. **Evaluation:** Thiết lập bộ benchmark 50 câu hỏi quy chế doanh nghiệp có ground truth, chạy đo lường tự động bằng **RAGAS** (Faithfulness, Answer Relevancy, Context Precision, Context Recall) trong pipeline CI/CD trước mỗi bản release. Bài lab đã chỉ ra hai bài học cần áp dụng ngay: (a) **ground truth phải mô tả đúng hành vi mong muốn** — test set của lab tính cả quy định cũ đã hết hiệu lực vào ground truth, khiến hệ thống đúng bị trừ điểm; (b) **điểm thấp phải được điều tra trước khi sửa hệ thống**, vì phần lớn điểm số thấp trong bài này đến từ bộ chấm chứ không phải từ RAG.

#### 3. Timeline triển khai
- **Tuần 1:** Chuẩn hóa dữ liệu quy chế nội bộ, tiền xử lý metadata (ngày ban hành, hiệu lực) và xây dựng parser Structure-Aware + Hierarchical Chunking.
- **Tuần 2:** Cài đặt Hybrid Search (BM25 underthesea + Qdrant BGE-M3) và tích hợp thuật toán RRF.
- **Tuần 3:** Triển khai tầng Cross-Encoder Reranker, kiểm thử độ trễ và tối ưu ONNX/Flashrank nếu cần.
- **Tuần 4:** Xây dựng bộ test set 50 câu hỏi, tích hợp RAGAS tự động đánh giá, hoàn thiện dashboard theo dõi chất lượng và đóng gói Docker.

#### 4. Rủi ro đã biết và cách xử lý
- **Context Recall suy giảm khi lọc phiên bản:** trong bài lab, việc loại bỏ quy định cũ làm Context Recall giảm 0.075. Với hệ thống thật, cần thống nhất chính sách với bộ phận pháp chế: trả lời theo quy định đang hiệu lực, và **lưu phiên bản đã trích dẫn** vào log để người dùng tự tra lịch sử khi cần.
- **Latency rerank trên CPU:** đo được 7 306,8 ms cho 20 ứng viên, chiếm 81,5% tổng thời gian. Bắt buộc phải đo lại trên phần cứng đích trước khi cam kết SLA; nếu chạy CPU, cần ONNX/Flashrank hoặc giảm số ứng viên.
- **Tài liệu dạng scan:** trong bài lab có 2 file PDF không có text layer bị bỏ qua hoàn toàn, khiến các câu hỏi về lương và phân loại dữ liệu phải suy luận từ nguồn khác. Cần pipeline OCR nếu doanh nghiệp có tài liệu dạng scan.
