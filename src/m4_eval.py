from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _zero_scores() -> dict:
    return {**{m: 0.0 for m in METRICS}, "per_question": []}


def _patch_faithfulness_unicode() -> None:
    """ragas 0.1.x serialize statements bằng json.dumps() mặc định (ensure_ascii=True) →
    judge nhận tiếng Việt dạng "\\u1ed7..." và hay chấm sai verdict. Ép ensure_ascii=False."""
    import functools
    import types
    import ragas.metrics._faithfulness as _faith

    _faith.json = types.SimpleNamespace(dumps=functools.partial(json.dumps, ensure_ascii=False),
                                        loads=json.loads)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # RAGAS cần OPENAI_API_KEY và Python 3.11+ → lỗi nào cũng trả về scores 0 thay vì crash pipeline.
    from config import EMBEDDING_MODEL, LLM_MAX_TOKENS, LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
    if not OPENAI_API_KEY:
        print("  ⚠️  RAGAS evaluation skipped: OPENAI_API_KEY chưa được cấu hình")
        return _zero_scores()
    try:
        import math
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        # Judge LLM qua endpoint OpenAI-compatible (OpenAI hoặc Ollama Cloud).
        # Ollama Cloud không có /v1/embeddings → embeddings cho answer_relevancy chạy local (bge-m3).
        llm = ChatOpenAI(model=LLM_MODEL, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY,
                         temperature=0, max_tokens=LLM_MAX_TOKENS)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                           encode_kwargs={"normalize_embeddings": True})
        answer_relevancy.strictness = 1  # endpoint không hỗ trợ n>1 completions
        _patch_faithfulness_unicode()
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        run_config = RunConfig(max_workers=4, timeout=300, max_retries=5)

        def _run(ds):
            return evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings,
                            raise_exceptions=False, run_config=run_config).to_pandas()

        df = _run(dataset)
        # Job lỗi/timeout (rate limit, JSON parse fail) → RAGAS trả NaN. Chạy lại các dòng NaN
        # thay vì coi là 0 — NaN là lỗi hạ tầng, không phải điểm chất lượng thật.
        for attempt in range(2):
            nan_rows = [i for i in range(len(df)) if df.loc[i, METRICS].isna().any()]
            if not nan_rows:
                break
            print(f"  ↻ Re-evaluating {len(nan_rows)} rows with NaN scores (attempt {attempt + 1})")
            retry_df = _run(dataset.select(nan_rows))
            for j, i in enumerate(nan_rows):
                for m in METRICS:
                    if math.isnan(df.loc[i, m]) and not math.isnan(retry_df.loc[j, m]):
                        df.loc[i, m] = retry_df.loc[j, m]

        def _score(value) -> float:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if math.isnan(value) else value

        per_question = [EvalResult(
            question=row["question"], answer=row["answer"],
            contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
            **{m: _score(row.get(m)) for m in METRICS})
            for _, row in df.iterrows()]

        # Aggregate giống RAGAS: trung bình trên các giá trị hợp lệ (bỏ NaN còn sót)
        aggregate = {m: float(df[m].mean(skipna=True)) if df[m].notna().any() else 0.0
                     for m in METRICS}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return _zero_scores()


# Diagnostic Tree: metric thấp nhất → nguyên nhân gốc + cách sửa
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating — answer chứa thông tin không có trong context",
                     "Tighten prompt (chỉ dùng context), lower temperature, trích dẫn nguồn"),
    "context_recall": ("Missing relevant chunks — retriever bỏ sót thông tin cần thiết",
                       "Improve chunking (parent-child), add BM25/hybrid, tăng top-k, query expansion"),
    "context_precision": ("Too many irrelevant chunks — context nhiễu, chunk liên quan bị xếp thấp",
                          "Add reranking or metadata filter (version/category), giảm top-k"),
    "answer_relevancy": ("Answer doesn't match question — trả lời lan man hoặc né tránh",
                         "Improve prompt template: trả lời trực tiếp, ngắn gọn, đúng trọng tâm"),
}

# Error Tree path (Output → Context → Query) tương ứng với từng metric
ERROR_TREE_PATH = {
    "faithfulness": "Output sai → Context đúng → Lỗi ở bước Generation",
    "answer_relevancy": "Output lệch câu hỏi → Context đúng → Lỗi ở Prompt/Generation",
    "context_precision": "Output thiếu chính xác → Context nhiễu → Lỗi ở Ranking/Reranking",
    "context_recall": "Output thiếu ý → Context thiếu → Query/Retrieval/Chunking chưa tốt",
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    analyzed = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        avg = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(avg, 4),
            "scores": {m: round(s, 4) for m, s in scores.items()},
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "error_tree": ERROR_TREE_PATH[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json",
                latency: dict | None = None):
    """Save evaluation report to JSON."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(r) if isinstance(r, EvalResult) else r
                         for r in results.get("per_question", [])],
    }
    if latency:
        report["latency"] = latency
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
