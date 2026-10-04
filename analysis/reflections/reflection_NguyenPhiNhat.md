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
| **RAGAS 4 Metrics & Diagnostic Tree** | M4 | `evaluate_ragas()`, `failure_analysis()` | Tự động đo lường 4 chỉ số cốt lõi: Faithfulness, Answer Relevancy, Context Precision, Context Recall. Cây chẩn đoán (Diagnostic Tree) tự động phân loại lỗi: nếu Context Recall thấp -> lỗi ở tầng chunking/retrieval; nếu Context Precision thấp -> thiếu reranking; nếu Faithfulness thấp -> LLM hallucination cần thắt chặt prompt. |
| **Contextual Prepend & Enrichment (Anthropic style)** | M5 | `contextual_prepend()`, `_enrich_single_call()`, `enrich_chunks()` | Dùng prompt tích hợp gọi LLM 1 lần duy nhất trên mỗi chunk để vừa tóm tắt (Summary), sinh 3 câu hỏi giả định (HyQA), viết câu ngữ cảnh nguồn (Contextual Prepend) và trích xuất Metadata. Giảm 49% lỗi retrieval theo nghiên cứu của Anthropic và tối ưu chi phí API. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. Lỗi xung đột Parent ID (Parent-Child Collision) giữa các tài liệu
- **Hiện tượng:** Khi chạy Production Pipeline ban đầu, gần như toàn bộ câu trả lời đều trả về `"Không tìm thấy."`. Điểm Faithfulness tụt xuống 0.28, Context Recall còn 0.43 mặc dù Naive Baseline đạt trên 0.83.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Trong `src/m1_chunking.py`, hàm `chunk_hierarchical()` đánh số `parent_id` cục bộ từ `parent_0`, `parent_1`,... cho mỗi văn bản.
  - Khi duyệt qua 26 tài liệu trong `src/pipeline.py`, dictionary `parent_lookup[pid] = parent.text` liên tục ghi đè key `"parent_0"`. Kết quả là `parent_lookup` chỉ lưu duy nhất đoạn cha của tài liệu cuối cùng (`28_chinh_sach_an_ninh_thong_tin.md`).
  - Khi câu hỏi về chế độ lương thưởng/nghỉ phép tìm trúng child `parent_0`, pipeline lại expand nhầm sang quy chế an ninh thông tin. LLM không thấy thông tin nên trả về "Không tìm thấy.".
- **Cách khắc phục:** Sửa `chunk_hierarchical` để prefix `parent_id` bằng tên nguồn tài liệu (`metadata.get("source")`), ví dụ: `01_quy_che_luong_thuong.md_parent_0`. Nhờ đó mỗi parent ID là duy nhất trên toàn hệ thống và không bị ghi đè.

### 2. Lỗi OpenRouter 402 ("requested up to 16384 tokens, but can only afford 7891")
- **Hiện tượng:** Khi gọi `client.chat.completions.create` trong `src/pipeline.py` và `naive_baseline.py`, API OpenRouter báo lỗi 402 khiến LLM generation thất bại và phải fallback lấy toàn bộ text context.
- **Nguyên nhân:** Khi không chỉ định `max_tokens`, client mặc định xin hạn mức tối đa của model (16k tokens), vượt quá ngân sách tạm giữ (credit hold) của tài khoản OpenRouter.
- **Cách khắc phục:** Bổ sung tham số `max_tokens=500` vào tất cả các lời gọi `chat.completions.create`. Model chỉ sinh câu trả lời ngắn gọn (1-2 đoạn), tránh lỗi 402 và tiết kiệm chi phí token.

### 3. Lỗi Timeout khi chạy kiểm thử tự động `check_lab.py`
- **Hiện tượng:** Chạy `python check_lab.py` bị lỗi timeout sau 120 giây ở bước `run_tests()`.
- **Nguyên nhân:** Mỗi unit test trong `test_m3.py` khởi tạo một đối tượng `CrossEncoderReranker()` mới, khiến mô hình `bge-reranker-v2-m3` nặng 2.2GB bị nạp lại từ ổ cứng 5 lần liên tiếp trên CPU, tiêu tốn hơn 60 giây.
- **Cách khắc phục:** Caching mô hình ở cấp module (`_CROSS_ENCODER_MODEL`), tải 1 lần dùng chung cho tất cả các test. Đồng thời thiết lập `max_length=256` khi gọi `model.predict()`, giúp tăng tốc độ suy luận Cross-Encoder trên CPU lên gấp 4 lần. Toàn bộ 37/37 unit tests chạy hoàn tất chỉ trong 25-30 giây.

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
5. **Evaluation:** Thiết lập bộ benchmark 50 câu hỏi quy chế doanh nghiệp có ground truth, chạy đo lường tự động bằng **RAGAS** (Faithfulness, Answer Relevancy, Context Precision, Context Recall) trong pipeline CI/CD trước mỗi bản release.

#### 3. Timeline triển khai
- **Tuần 1:** Chuẩn hóa dữ liệu quy chế nội bộ, tiền xử lý metadata (ngày ban hành, hiệu lực) và xây dựng parser Structure-Aware + Hierarchical Chunking.
- **Tuần 2:** Cài đặt Hybrid Search (BM25 underthesea + Qdrant BGE-M3) và tích hợp thuật toán RRF.
- **Tuần 3:** Triển khai tầng Cross-Encoder Reranker, kiểm thử độ trễ (latency < 150ms) và tối ưu ONNX/Flashrank nếu cần.
- **Tuần 4:** Xây dựng bộ test set 50 câu hỏi, tích hợp RAGAS tự động đánh giá, hoàn thiện dashboard theo dõi chất lượng và đóng gói Docker.
