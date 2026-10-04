# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Phi Nhật  
**Mã số học viên (MSSV):** 2A202602658  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production (Trước fix) | Δ |
|--------|---------------|-------------------------|---|
| Faithfulness | 0.8308 | 0.2875 | -0.5433 |
| Answer Relevancy | 0.5211 | 0.2780 | -0.2431 |
| Context Precision | 0.9250 | 0.3833 | -0.5417 |
| Context Recall | 0.9250 | 0.4333 | -0.4917 |

*Nhận xét ban đầu:* Hệ thống Production trước khi sửa lỗi có điểm số sụt giảm nghiêm trọng so với Naive Baseline. Nguyên nhân chủ đạo bắt nguồn từ lỗi thiết kế liên kết phân cấp cha - con (Parent-Child Collision) khiến tài liệu truy xuất bị tráo đổi hoàn toàn sang tài liệu an ninh mạng, khiến LLM từ chối trả lời ("Không tìm thấy.") và kéo tụt toàn bộ 4 chỉ số RAGAS.

---

## Bottom-5 Failures

### #1
- **Question:** Phụ cấp ăn trưa hàng tháng là bao nhiêu?
- **Expected:** Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng, chi trả cùng kỳ lương.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.0), Context Recall (0.0)
- **Error Tree:** Output sai ("Không tìm thấy.") → Context đúng? KHÔNG (Context trích xuất bị tráo sang tài liệu An ninh thông tin) → Query OK? CÓ (câu hỏi rõ nghĩa) → Fix ở bước: **M1 (Hierarchical Chunking) & Pipeline Parent Expansion**.
- **Root cause:** Trong `chunk_hierarchical()`, các đoạn cha đều được đánh mã cục bộ `parent_0`, `parent_1`,... Khi duyệt qua 26 file văn bản, `parent_lookup` liên tục ghi đè key `"parent_0"`. Kết quả là khi child chunk của quy chế lương thưởng khớp truy vấn, pipeline lại lấy parent chunk của file cuối cùng (`28_chinh_sach_an_ninh_thong_tin.md`). LLM không tìm thấy thông tin lương thưởng trong context an ninh nên trả về "Không tìm thấy.".
- **Suggested fix:** Thêm prefix tên tài liệu vào `parent_id` (ví dụ `01_quy_che_luong_thuong.md_parent_0`) trong `src/m1_chunking.py` để đảm bảo mỗi parent chunk có định danh duy nhất trên toàn kho tài liệu.

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.0), Context Precision (0.0)
- **Error Tree:** Output sai → Context đúng? KHÔNG (Bị mất đoạn quy định thời hạn tạm ứng 15 ngày và lãi phạt 2%/tháng) → Retrieval bỏ sót → Fix ở bước: **M1 & M2 (Hybrid Search)**.
- **Root cause:** Xung đột Parent ID tương tự ca #1 làm mất ngữ cảnh quy chế tài chính. Thêm vào đó, câu hỏi dạng suy luận nhiều bước (tính số ngày trễ = 20 - 15 = 5 ngày, tính số tiền phạt) yêu cầu ngữ cảnh đầy đủ của cả Điều khoản Tạm ứng và Chế tài.
- **Suggested fix:** Sau khi sửa Parent ID, kết hợp BM25 bắt từ khóa chính xác "tạm ứng", "thời hạn thanh toán" và cấu hình prompt LLM yêu cầu thực hiện bước tính toán trung gian (Chain-of-Thought).

### #3
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected:** Hạn mức bảo hiểm sức khỏe PVI cho nhân viên là 200.000.000 VNĐ/năm, bao gồm nội trú, ngoại trú và nha khoa.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.0), Context Recall (0.0)
- **Error Tree:** Output sai → Context đúng? KHÔNG → Retrieval thiếu thực thể "PVI" → Fix ở bước: **M1 & M2**.
- **Root cause:** Parent expansion trả về tài liệu sai chủ đề. Ngoài ra từ khóa "PVI" là tên riêng thương hiệu viết tắt; nếu Dense Search đơn thuần không được bổ sung từ vựng BM25 thì điểm tương đồng vector thấp.
- **Suggested fix:** Đảm bảo `segment_vietnamese()` xử lý tốt các từ viết hoa/từ viết tắt tiếng Việt, giữ nguyên token "PVI" và đưa đoạn cha của tài liệu Phúc lợi y tế vào context.

### #4
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.0), Context Recall (0.0)
- **Error Tree:** Output sai → Context đúng? KHÔNG (Thiếu bảng ma trận thẩm quyền phê duyệt nghỉ phép) → Fix ở bước: **M1 (Structure-Aware Chunking) & M3 (Reranker)**.
- **Root cause:** Bảng thẩm quyền phê duyệt được chia theo các khung: 1-3 ngày (Trưởng phòng), 4-15 ngày (Giám đốc khối), 16-30 ngày (CEO). Cắt đoạn ngắt giữa bảng làm mất liên kết giữa mốc "20 ngày" và "CEO".
- **Suggested fix:** Cắt đoạn theo tiêu đề mục (Structure-Aware Chunking) để giữ trọn vẹn toàn bộ bảng biểu phân quyền, tránh tách rời các mốc ngày.

### #5
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Got:** Mật khẩu phải có tối thiểu **8 ký tự**.
- **Worst metric:** Faithfulness (0.0), Context Precision (0.0)
- **Error Tree:** Output sai (Lấy quy định cũ đã hết hiệu lực) → Context đúng? KHÔNG ĐỦ CHÍNH XÁC (Context chứa cả v1.0 và v2.0 nhưng v1.0 xếp trước) → Fix ở bước: **M5 (Metadata Enrichment) & System Prompt**.
- **Root cause:** Thử thách xung đột phiên bản (Version Conflict). Trong dữ liệu có cả `05_chinh_sach_mat_khau_v1.0.md` (8 ký tự) và `06_chinh_sach_mat_khau_v2.0.md` (12 ký tự). Cả 2 đều chứa từ khóa "mật khẩu tối thiểu", nhưng v1.0 được BM25 đẩy lên top đầu. LLM thấy context v1.0 đứng trước nên đã tin tưởng và trả lời 8 ký tự.
- **Suggested fix:** 
  1. Trích xuất metadata `version` và `effective_date` ở M5 để áp dụng bộ lọc Metadata Filtering (chỉ tìm tài liệu còn hiệu lực).
  2. Bổ sung quy tắc trong System Prompt: "Nếu gặp các văn bản có phiên bản khác nhau (v1.0 vs v2.0), luôn ưu tiên áp dụng phiên bản có số hiệu cao nhất hoặc mới nhất".

---

## Case Study (cho presentation)

**Question chọn phân tích:** *Mật khẩu phải có tối thiểu bao nhiêu ký tự?* (Ca xung đột phiên bản tài liệu — Versioning Dilemma).

**Error Tree walkthrough:**
1. **Output đúng?** → KHÔNG. Mô hình trả lời 8 ký tự (theo bản cũ v1.0), trong khi quy định có hiệu lực v2.0 bắt buộc 12 ký tự.
2. **Context đúng?** → MỘT NỬA. Cả hai đoạn trích từ v1.0 và v2.0 đều được Hybrid Search tìm thấy, nhưng đoạn v1.0 lại đứng ở vị trí rank=0 do trùng khớp từ khóa cao hơn.
3. **Query rewrite OK?** → Query người dùng không nêu rõ năm/phiên bản ("Mật khẩu phải có tối thiểu bao nhiêu ký tự?").
4. **Fix ở bước:**
   - **Tầng Data / Ingestion (M5):** Tự động gắn tag `version: "v2.0"`, `is_active: true` vào metadata chunk. Khi index vào Qdrant, đặt payload filter `is_active == true` hoặc hạ điểm các tài liệu archive/deprecated.
   - **Tầng Prompt Engineering:** Bổ sung ràng buộc giải quyết xung đột vào system prompt: *"Khi xuất hiện nhiều chính sách xung đột nhau theo thời gian, ưu tiên áp dụng phiên bản mới nhất và ghi chú rõ phiên bản được trích dẫn."*

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Triển khai Metadata Filtering động:** Tự động phát hiện trường metadata `status` hoặc `version` từ tên file (regex `_v\d+\.\d+`) để chỉ cho phép Hybrid Search truy vấn trên các văn bản hiện hành (Active).
- **Áp dụng Query Expansion / HyDE (Hypothetical Document Embeddings):** Sinh trước câu trả lời giả định hoặc sinh các từ đồng nghĩa pháp lý trước khi tìm kiếm để tăng độ bao phủ Context Recall.
- **Fine-tune Cross-Encoder Reranker:** Tinh chỉnh mô hình Reranker trên bộ dữ liệu câu hỏi - điều khoản nội bộ tiếng Việt để nhận biết tốt hơn tính thứ bậc của văn bản pháp quy.
