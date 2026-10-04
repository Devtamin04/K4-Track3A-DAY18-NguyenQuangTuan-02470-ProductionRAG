from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import LLM_MAX_TOKENS, LLM_MODEL, OPENAI_BASE_URL, RERANK_TOP_K

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về chính sách nội bộ công ty.
Quy tắc:
1. CHỈ dùng thông tin trong context. Không suy đoán, không thêm kiến thức bên ngoài.
2. Trả lời trực tiếp vào câu hỏi ngay câu đầu tiên, nêu rõ con số/điều kiện/người phê duyệt cụ thể.
   Giữ nguyên đối tượng và điều kiện áp dụng như context ghi (VD: "nhân viên chính thức", "từ 3 năm trở lên"), không khái quát hóa.
3. Nếu có nhiều phiên bản chính sách mâu thuẫn, dùng phiên bản hiện hành (mới nhất) và ghi chú ngắn rằng phiên bản cũ đã bị thay thế.
4. Với câu hỏi cần tính toán hoặc kết hợp nhiều tài liệu: nêu quy định áp dụng (diễn đạt sát nguyên văn context, kèm con số gốc), rồi tính toán từng bước và kết luận.
5. Chỉ trả lời đúng phạm vi câu hỏi, không thêm thông tin phụ không được hỏi. Ngắn gọn (1-4 câu). Nếu context không có thông tin → trả lời "Không tìm thấy thông tin trong tài liệu."."""

# Latency breakdown (giây) — điền trong build_pipeline / evaluate_pipeline
LATENCY: dict = {}


def _doc_header(text: str) -> str:
    """Tiêu đề (# ...) + dòng phiên bản (> ...) đầu tài liệu — gắn vào parent để giữ ngữ cảnh version."""
    lines = []
    for line in text.splitlines()[:3]:
        if line.startswith(("# ", ">")):
            lines.append(line.strip())
    return "\n".join(lines)


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1) — hierarchical: index child (precision), trả về parent (context)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        header = _doc_header(doc["text"])
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_texts = {}
        for p in parents:
            text = p.text if not header or p.text.startswith(header.split("\n")[0]) else f"{header}\n\n{p.text}"
            parent_texts[p.metadata["parent_id"]] = text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {
                **child.metadata, "parent_id": child.parent_id,
                "doc_header": header, "parent_text": parent_texts[child.parent_id]}})
    LATENCY["chunking_s"] = round(time.time() - t0, 2)
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5) — combined mode: 1 API call/chunk
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        # Index = context line + chunk + hypothesis questions (bridge vocabulary gap query ↔ document)
        all_chunks = [{"text": "\n".join([e.enriched_text, *e.hypothesis_questions]),
                       "metadata": {**e.auto_metadata, "original_text": e.original_text}}
                      for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    LATENCY["enrichment_s"] = round(time.time() - t0, 2)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    LATENCY["indexing_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    LATENCY["reranker_load_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def retrieve(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
             timings: dict | None = None) -> list[str]:
    """Hybrid search top-20 children → rerank → top-k parent contexts (dedup theo parent_id)."""
    t0 = time.perf_counter()
    results = search.search(query)
    t1 = time.perf_counter()
    # Rerank trên text gốc của child (không kèm context line + HyQA) → ngắn hơn, nhanh và sát nội dung hơn
    docs = [{"text": r.metadata.get("original_text") or r.text, "score": r.score, "metadata": r.metadata}
            for r in results]
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    t2 = time.perf_counter()
    if timings is not None:
        timings["search_ms"] = (t1 - t0) * 1000
        timings["rerank_ms"] = (t2 - t1) * 1000

    contexts, seen = [], set()
    for r in reranked or results:
        pid = r.metadata.get("parent_id") or r.text
        if pid in seen:
            continue
        seen.add(pid)
        contexts.append(r.metadata.get("parent_text") or r.text)
        if len(contexts) == RERANK_TOP_K:
            break
    return contexts


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              timings: dict | None = None) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    contexts = retrieve(query, search, reranker, timings)

    t0 = time.perf_counter()
    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=OPENAI_BASE_URL)
            context_str = "\n\n---\n\n".join(f"[Tài liệu {i+1}]\n{c}" for i, c in enumerate(contexts))
            resp = client.chat.completions.create(model=LLM_MODEL, temperature=0, max_tokens=LLM_MAX_TOKENS, messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    if timings is not None:
        timings["llm_ms"] = (time.perf_counter() - t0) * 1000
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    query_timings = []

    for i, item in enumerate(test_set):
        timings = {}
        answer, contexts = run_query(item["question"], search, reranker, timings)
        query_timings.append(timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    LATENCY["ragas_eval_s"] = round(time.time() - t0, 2)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    # Latency breakdown per query (avg / p95 / max, ms)
    per_query = {}
    for step in ["search_ms", "rerank_ms", "llm_ms"]:
        vals = sorted(t.get(step, 0.0) for t in query_timings)
        if vals:
            per_query[step] = {"avg": round(sum(vals) / len(vals), 1),
                               "p95": round(vals[min(len(vals) - 1, int(0.95 * len(vals)))], 1),
                               "max": round(vals[-1], 1)}
    LATENCY["per_query_ms"] = per_query
    print("\nLATENCY BREAKDOWN (per query, ms)")
    print(f"  {'Step':<12} {'avg':>8} {'p95':>8} {'max':>8}")
    for step, s in per_query.items():
        print(f"  {step[:-3]:<12} {s['avg']:>8.1f} {s['p95']:>8.1f} {s['max']:>8.1f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures, latency=dict(LATENCY))
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
