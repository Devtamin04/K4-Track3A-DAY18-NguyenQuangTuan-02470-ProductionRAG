# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyen Quang Tuan (02470)
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Với threshold 0.85 (dùng `all-MiniLM-L6-v2`), cả corpus bị cắt thành **208 chunks** (avg 99 chars, min 6) so với basic **51 chunks** (avg 410). Hạ threshold xuống 0.7 vẫn ra 182 chunks, còn 0.5 thì gộp lại còn 29. Model tiếng Anh cho cosine giữa các câu tiếng Việt thấp nên 0.85 quá chặt, cắt vụn cả ý. Vì vậy pipeline production dùng **hierarchical** chứ không dùng semantic. |
| Hierarchical (parent-child) | M1 | `chunk_hierarchical()` + `retrieve()` trong `pipeline.py` | 113 children (≤ 256 chars) dùng để index, còn context trả về là parent (≤ 2048, có kèm tiêu đề + dòng phiên bản). Context recall tăng từ **0.754 lên 0.875**. Đã thêm prefix `source::` vào `parent_id` vì nếu không, `parent_0` của 26 tài liệu sẽ trùng nhau. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks, mỗi chunk là 1 section `##` kèm `metadata["section"]`. Corpus này có cấu trúc markdown rất đều nên đây là phương án thay thế tốt; max 788 chars vì section chứa bảng được giữ nguyên, không bị cắt. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search`, `reciprocal_rank_fusion()` | RRF (k=60) chỉ dùng thứ hạng nên không cần chuẩn hóa thang điểm BM25 (0–20) với cosine (0–1). BM25 bắt được các token chính xác như "PVI", "MFA", "50.000.000", còn dense bắt các câu hỏi diễn đạt khác đi. Phải `replace("_", " ")` sau `word_tokenize`, nếu không thì "nghỉ_phép" trong tài liệu sẽ không khớp "nghỉ phép" trong query. Search latency avg **80 ms**. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` trên CPU: ban đầu **17.9 s/query** (20 candidates là text đã enrich kèm HyQA); đổi sang rerank text gốc của child thì còn **7.5 s** (−58%). Context precision đạt **0.94**, nhưng vẫn nhầm "tạm ứng" với "mua sắm" ở câu 55 triệu (cả hai đều có bảng ngưỡng tiền). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Faithfulness thấp nhất (0.79) vì judge NLI chấm chặt: (1) "nhân viên" so với "nhân viên **chính thức**" bị chấm 0; (2) các bước tính toán và dòng trong bảng markdown khó được judge xác nhận. Judge `gpt-oss:120b` dao động ±0.1 giữa các lần chạy, nên cần chạy nhiều lần. |
| Contextual embeddings | M5 | `_enrich_single_call()` (combined, 1 call/chunk) | Một call trả về context line ("đoạn này thuộc Chính sách nghỉ phép năm v2024, đang hiệu lực…") + 3 câu hỏi HyQA + metadata (category, version). Context line giúp child 256 chars vẫn mang được thông tin tài liệu/phiên bản; HyQA thu hẹp khoảng cách từ vựng giữa câu hỏi người dùng và văn bản quy định. Kết quả được cache theo hash, nên lần chạy sau enrichment chỉ mất 14 s thay vì 216 s. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi 1: Ollama Cloud không có endpoint embeddings**
  - Exact error: `{"error":"path \"/v1/embeddings\" not found"}`
  - Debug: dùng `curl` thử từng endpoint: `/v1/chat/completions` hoạt động (kể cả `response_format: json_object`), còn `/v1/embeddings` thì không có.
  - Giải quyết: tách cấu hình. Judge LLM gọi qua endpoint OpenAI-compatible (`OPENAI_BASE_URL`, `LLM_MODEL` trong `.env`), còn embeddings cho `answer_relevancy` chạy local bằng `HuggingFaceEmbeddings(BAAI/bge-m3)`.

- **Lỗi 2: Faithfulness = 0.0 cho một câu trả lời rõ ràng là đúng**
  - Hiện tượng: smoke test `answer="Nhân viên được nghỉ 15 ngày phép năm."`, context "Nhân viên **chính thức** được nghỉ phép năm 15 ngày" → `{'faithfulness': 0.0000}`.
  - Debug: gắn `BaseCallbackHandler` để in prompt/output của judge. Lý do judge đưa ra: *"does not say that all employees receive this benefit"*. Vậy metric không lỗi mà judge chấm rất chặt.
  - Giải quyết: thêm vào system prompt quy tắc "giữ nguyên đối tượng và điều kiện áp dụng như context ghi, không khái quát hóa".

- **Lỗi 3: Statements tiếng Việt bị gửi cho judge dưới dạng escape `\u1ed7…`**
  - Phát hiện khi trace: `statements: "[\"M\\u1ed7i nh\\u00e2n vi\\u00ean…\"]"`. Nguyên nhân: `ragas/metrics/_faithfulness.py:203` gọi `json.dumps(statements)` mà không có `ensure_ascii=False`.
  - Giải quyết: monkeypatch trong `_patch_faithfulness_unicode()` (không sửa site-packages).

- **Lỗi 4: Điểm RAGAS dao động mạnh giữa các lần chạy**
  - Faithfulness 0.75 (run 1) → 0.70 (run 2). Chấm lại đúng bộ câu trả lời của run 2 thì ra 0.82. Câu "nghỉ không lương 20 ngày cần CEO duyệt" được 1.0, lần sau ra 0.0.
  - Giả thuyết ban đầu là job bị lỗi trả NaN rồi bị tính thành 0. Tôi đã thêm bước re-evaluate các dòng NaN và tính aggregate bằng `mean(skipna=True)`, nhưng log run cuối không có dòng NaN nào. Kết luận: nguồn nhiễu là **judge không deterministic**, không phải hạ tầng.
  - Bài học: muốn so sánh hai cấu hình RAG thì cần nhiều lần chạy hoặc judge mạnh hơn; chênh lệch < 0.05 trên 20 câu hỏi không có ý nghĩa thống kê.

- **Lỗi 5: Docker daemon không chạy**
  - Exact error: `failed to connect to the docker API at unix://~/.docker/desktop/docker.sock … no such file or directory`
  - Giải quyết: `DenseSearch` đã có fallback `QdrantClient(":memory:")`; dùng `delete_collection` + `create_collection` thay cho `recreate_collection` (đã deprecated).

- **Kiến thức còn thiếu & cách bổ sung:** cách RAGAS tính từng metric (statement extraction → NLI verdict; noncommittal làm answer_relevancy = 0; context_precision là average precision theo thứ hạng). Tôi bổ sung bằng cách đọc source `ragas/metrics/*.py` và trace prompt thực tế thay vì chỉ nhìn điểm tổng.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Chatbot hỏi đáp tài liệu nội bộ (quy định, chính sách, hướng dẫn) tiếng Việt

#### 1. Hiện trạng
- **Pipeline hiện tại:** fixed-size chunking 500 chars → dense embedding → top-3 → LLM (giống naive baseline của lab).
- **Vấn đề / Bottlenecks:** trả lời theo phiên bản chính sách cũ; câu hỏi multi-hop chỉ trả lời được một nửa; không có bộ đo chất lượng nên không biết thay đổi nào thực sự cải thiện.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** hierarchical (child 256 để index, parent ≈ section/tài liệu làm context) kết hợp structure-aware cho tài liệu markdown/có heading. Trong lab, cách này tăng recall +0.12. Semantic chunking chỉ dùng nếu có embedding model tiếng Việt tốt và đã tune threshold.
2. **Search retrieval:** Hybrid BM25 (underthesea) + dense bge-m3 + RRF. Văn bản quy định có rất nhiều con số và mã định danh mà dense hay bỏ sót. Thêm metadata filter `version/effective_date` và quan hệ `supersedes` giữa các phiên bản.
3. **Reranking:** có, dùng `bge-reranker-v2-m3` nhưng chạy GPU hoặc ONNX, chỉ rerank top-10, mục tiêu < 300 ms/query. Phương án dự phòng là flashrank multilingual nếu chỉ có CPU.
4. **Evaluation:** RAGAS 4 metrics trên golden set khoảng 50 câu chia theo loại (lookup, version, negation, multi-hop, numeric); chạy 3 lần và lấy trung bình; thêm metric exact-match cho câu numeric. Gắn vào CI để mỗi thay đổi pipeline đều có số liệu so sánh.
5. **Enrichment:** combined single-call (context line + HyQA + metadata) có cache theo hash; chuyển bảng markdown thành câu trước khi embed; thêm query decomposition cho câu multi-hop.

#### 3. Timeline triển khai
- **Tuần 1:** Xây golden set 50 câu và đo baseline RAGAS (3 lần chạy).
- **Tuần 2:** Hierarchical + structure-aware chunking, hybrid BM25 + dense + RRF; đo lại.
- **Tuần 3:** Reranker (GPU/ONNX) + metadata version filter + liên kết `supersedes`.
- **Tuần 4:** Enrichment combined + cache, query decomposition multi-hop; đưa RAGAS vào CI và theo dõi latency p95.
