# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyen Quang Tuan (02470)
**Khóa:** K4 - Track 3A

---

## Setup

| Thành phần | Naive Baseline | Production |
|---|---|---|
| Chunking | Paragraph (`chunk_basic`, 500 chars) → 57 chunks | Hierarchical parent 2048 / child 256 → 113 children; trả về **parent** làm context |
| Enrichment | — | Combined single-call (`_enrich_single_call`): context line + 3 HyQA questions + metadata |
| Retrieval | Dense (bge-m3) top-3 | BM25 (underthesea) + Dense (bge-m3) → RRF top-20 |
| Rerank | — | `bge-reranker-v2-m3` trên child gốc → dedup theo `parent_id` → top-3 parent |
| Generator | `gpt-oss:120b` (Ollama Cloud), prompt cơ bản | `gpt-oss:120b`, prompt version-aware + "giữ nguyên điều kiện như context" |
| Judge (RAGAS 0.1.22) | `gpt-oss:120b` + embeddings bge-m3 local | như bên trái |

## RAGAS Scores (run cuối, `reports/ragas_report.json`)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6419 | **0.7879** | +0.1460 |
| Answer Relevancy | 0.8327 | **0.8022** | −0.0305 |
| Context Precision | 0.9417 | **0.9417** | 0.0000 |
| Context Recall | 0.7542 | **0.8750** | +0.1208 |

Cả 4 metric production ≥ 0.75. Cải thiện lớn nhất là **faithfulness** (+0.15) và **context recall** (+0.12): parent-chunk mang đủ ngữ cảnh (tiêu đề + phiên bản + toàn bộ điều khoản), còn BM25 bắt đúng các từ khóa/con số mà dense bỏ sót.

**Lưu ý về độ nhiễu của judge:** trong 3 lần chạy production, faithfulness dao động 0.70 → 0.79; chấm lại *cùng một bộ câu trả lời* hai lần cho 0.70 rồi 0.82. Vì vậy chênh lệch < ~0.05 (VD: answer relevancy −0.03) nằm trong biên nhiễu của judge, không nên kết luận là bị suy giảm.

## Latency breakdown (CPU, không GPU)

| Bước | Thời gian |
|---|---|
| Chunking (26 docs) | 0.02 s |
| Enrichment 113 chunks | ~216 s lần đầu, ~14 s khi đã cache |
| Indexing (BM25 + bge-m3 encode) | ~76–91 s |
| Load reranker | ~6 s |
| **Per query** — hybrid search | avg 80 ms · p95 98 ms |
| **Per query** — rerank 20 candidates | avg 7.5 s · p95 10.3 s (trước tối ưu: 17.9 s) |
| **Per query** — LLM generation | avg 1.2 s · p95 2.2 s |
| RAGAS eval (20 q × 4 metrics) | ~210 s |

Nút thắt là cross-encoder 568M tham số chạy trên CPU. Rerank trên text gốc của child (thay cho text đã enrich kèm HyQA) giảm latency ~58%.

## Bottom-5 Failures

### #1
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Cần phê duyệt bởi Tổng Giám đốc (CEO), vì giá trị đơn hàng trên 50.000.000 VNĐ." — **đúng**.
- **Worst metric:** faithfulness = 0.00 (context_precision = 0.50)
- **Error Tree:** Output đúng → Context đúng (có `mua_sam.md`) nhưng **xếp hạng 2**, sau `tam_ung.md` → Query OK → lỗi nằm ở **Reranking** và ở **judge**.
- **Root cause:** (1) Câu trả lời lấy từ một dòng trong *bảng markdown* (`| Trên **50.000.000 VNĐ** | Tổng Giám đốc (CEO) |`); judge NLI không ghép được hàng bảng với statement, nên chấm 0 dù câu trả lời đúng. (2) Reranker xếp chính sách tạm ứng (cũng có ngưỡng tiền và người duyệt) lên trên quy trình mua sắm.
- **Suggested fix:** Khi enrich, chuyển bảng thành câu ("Đơn hàng trên 50 triệu cần CEO phê duyệt"); thêm metadata filter `category` (mua sắm ≠ tạm ứng) trước khi rerank.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (v2024); lương Senior (P3-P4) 20–35 triệu VNĐ/tháng.
- **Got:** Tính đúng 18 ngày, nhưng trả lời "không tìm thấy thông tin về mức lương".
- **Worst metric:** answer_relevancy = 0.00 (context_recall = 0.50)
- **Error Tree:** Output thiếu ý → Context **thiếu** `bang_luong_2024.md` → Query multi-hop chưa được tách → lỗi ở **Retrieval (query)**.
- **Root cause:** Câu hỏi multi-hop ghép hai chủ đề; vế "nghỉ phép năm" chiếm phần lớn lexical/semantic signal, nên cả 3 slot top-3 rơi vào các tài liệu nghỉ phép (v2024, v2023, không lương). Câu trả lời có cụm "không thể trả lời" nên RAGAS đánh dấu *noncommittal* và cho answer_relevancy = 0.
- **Suggested fix:** Query decomposition (LLM tách thành 2 sub-query) rồi retrieve cho từng sub-query và merge bằng RRF; hoặc đảm bảo đa dạng nguồn (MMR / tối đa 1 parent mỗi chủ đề).

### #3
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày, quá 5 ngày, phí 2%/tháng × 15 triệu = 300.000 VNĐ/tháng → ~50.000 VNĐ cho 5 ngày.
- **Got:** Nêu đúng hạn 15 ngày và phí 2%/tháng, tính ra ~50.000 VNĐ.
- **Worst metric:** context_recall = 0.33 (faithfulness = 0.40)
- **Error Tree:** Output đúng → Context đúng (`tam_ung.md` rank 1) → Query OK → lỗi do **metric/judge**, không phải pipeline.
- **Root cause:** Ground truth có các giá trị *suy ra* (300.000 VNĐ/tháng, pro-rata 50.000 VNĐ) không có nguyên văn trong context nên recall bị trừ; tương tự, các bước tính toán trong câu trả lời bị judge coi là "không suy ra trực tiếp được từ context" nên faithfulness thấp.
- **Suggested fix:** Với câu hỏi numeric, viết ground truth tách phần "quy định" và phần "tính toán"; có thể thêm metric riêng cho tính toán (so khớp đáp số) thay vì chỉ dùng faithfulness NLI.

### #4
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc ≥ 1 năm; nghỉ sau 8 tháng → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "Phải hoàn trả 100% chi phí đào tạo, tức 25.000.000 VNĐ, vì nghỉ việc trước thời hạn cam kết 1 năm." — **đúng, sát nguyên văn**.
- **Worst metric:** faithfulness = 0.33 (context_recall = 0.50)
- **Error Tree:** Output đúng → Context đúng (`hoan_chi_dao_tao.md` rank 1) → Query OK → lỗi ở **judge (Evaluation)**.
- **Root cause:** Context chứa nguyên văn "cam kết làm việc ít nhất 1 năm… hoàn trả 100% chi phí". Ở các lần chạy khác, cùng loại câu trả lời này được chấm 0.0 rồi 0.33: **judge `gpt-oss:120b` không ổn định** (statement bị dịch sang tiếng Anh rồi mới NLI). Đây là failure của evaluation, không phải của RAG.
- **Suggested fix:** Dùng judge ổn định hơn (GPT-4o / Claude) hoặc chạy RAGAS nhiều lần rồi lấy trung bình; cố định prompt NLI tiếng Việt.

### #5
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có — v2.0 bắt buộc MFA cho email, VPN, hệ thống nội bộ; v1.0 cũ không yêu cầu MFA.
- **Got:** "Có. Theo Chính sách mật khẩu v2.0, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ."
- **Worst metric:** context_recall = 0.50 (answer_relevancy = 0.61)
- **Error Tree:** Output đúng nhưng thiếu ý so sánh version → Context thiếu `mat_khau_v1.md` → Query OK → lỗi ở **Retrieval (version linking)**.
- **Root cause:** Đây là câu hỏi dạng *negation/version*: v1.0 không chứa từ "MFA" nên cả BM25 lẫn dense đều không thể kéo v1.0 lên. Thông tin "v1 không có MFA" chỉ suy ra được khi có cả hai phiên bản trong context.
- **Suggested fix:** Lưu quan hệ `supersedes` trong metadata (v2.0 "thay thế Chính sách mật khẩu v1.0"); khi retrieve được phiên bản hiện hành thì tự động kéo thêm phiên bản bị thay thế làm context phụ.

## Case Study (cho presentation)

**Question chọn phân tích:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?" (multi-hop)

**Error Tree walkthrough:**
1. Output đúng? → **Một nửa**: 18 ngày phép đúng (đã áp dụng đúng v2024 thay vì v2023), nhưng phần lương thì trả lời "không tìm thấy".
2. Context đúng? → **Không đủ**: top-3 parent là nghỉ phép v2024, nghỉ không lương, nghỉ phép v2023; thiếu `bang_luong_2024.md` (có dòng `Senior (P3-P4) | 20.000.000 - 35.000.000`). Generator đã làm đúng khi không bịa ra mức lương, nên faithfulness vẫn đạt 0.875.
3. Query rewrite OK? → **Không**: pipeline dùng nguyên câu hỏi gồm hai ý; không có bước decomposition.
4. Fix ở bước: **Query transformation + Retrieval**: tách sub-query ("số ngày phép năm với 9 năm thâm niên" và "khoảng lương Senior"), retrieve từng sub-query, merge bằng RRF và giữ ít nhất 1 parent cho mỗi sub-query.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu multi-hop, và liên kết `supersedes` giữa các phiên bản (fix #2, #5).
- Chuyển bảng markdown thành câu trong bước enrichment (fix #1).
- Chạy RAGAS 3 lần rồi lấy trung bình, hoặc đổi sang judge mạnh hơn, để giảm nhiễu (#3, #4).
- Giảm latency rerank: chạy reranker trên GPU/ONNX, hoặc chỉ rerank top-10 thay vì top-20.
