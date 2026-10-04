from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import hashlib, json, os, re, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MAX_TOKENS, LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           ".cache", "enrichment.json")
_client = None


def _get_client():
    """Lazy OpenAI client (dùng chung cho mọi technique)."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def _chat(system: str, user: str, json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=LLM_MAX_TOKENS,
        temperature=0,
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            return _chat("Tóm tắt đoạn văn sau trong 1-2 câu ngắn gọn bằng tiếng Việt, "
                         "giữ nguyên các con số quan trọng.", text)
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    # Extractive fallback (không cần API): 2 câu đầu
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.replace("\n", " ")) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể "
                            "trả lời. Trả về mỗi câu hỏi trên 1 dòng, kết thúc bằng dấu '?', không đánh số.",
                            text)
            questions = [q.strip().lstrip("0123456789.-) ").strip() for q in content.split("\n")]
            return [q for q in questions if q][:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    # Extractive fallback: biến câu khẳng định thành câu hỏi
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY:
        try:
            context = _chat("Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về "
                            "chủ đề gì. Chỉ trả về 1 câu.",
                            f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}")
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    # Simple fallback: tên tài liệu làm context
    prefix = f"Trích từ {document_title}.\n\n" if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


_METADATA_SCHEMA = ('{"topic": "...", "entities": ["..."], '
                    '"category": "leave|salary|it|workflow|training|admin|safety|compliance", '
                    '"version": "phiên bản nếu có, ngược lại rỗng", "language": "vi|en"}')


def _fallback_metadata(text: str) -> dict:
    version = re.search(r'Phiên bản:\s*([\w.]+)', text)
    return {"topic": "general", "entities": [], "category": "policy",
            "version": version.group(1) if version else "", "language": "vi"}


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(f"Trích xuất metadata từ đoạn văn. Trả về JSON: {_METADATA_SCHEMA}",
                            text, json_mode=True)
            return json.loads(content)
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


_COMBINED_PROMPT = f"""Bạn là trợ lý chuẩn bị dữ liệu cho hệ thống tìm kiếm tài liệu nội bộ.
Phân tích đoạn văn (trích từ tài liệu đã cho) và trả về JSON:
{{
  "summary": "tóm tắt 1-2 câu, giữ nguyên con số quan trọng",
  "questions": ["3 câu hỏi tiếng Việt mà đoạn văn trả lời được"],
  "context": "1 câu mô tả đoạn văn thuộc tài liệu/chính sách nào (kèm phiên bản, còn hiệu lực hay đã bị thay thế nếu biết) và nói về chủ đề gì",
  "metadata": {_METADATA_SCHEMA}
}}"""


def _load_cache() -> dict:
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def _enrich_single_call(text: str, source: str, doc_header: str = "") -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    Kết quả được cache theo hash(nội dung) → chạy lại pipeline không tốn thêm API call.
    """
    if not OPENAI_API_KEY:
        return {}
    key = hashlib.sha256(f"{LLM_MODEL}|{source}|{doc_header}|{text}".encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]
    try:
        doc_info = f"Tài liệu: {source}\n{doc_header}".strip()
        content = _chat(_COMBINED_PROMPT, f"{doc_info}\n\nĐoạn văn:\n{text}", json_mode=True)
        result = json.loads(content)
        cache[key] = result
        _save_cache(cache)
        return result
    except Exception as e:
        print(f"  ⚠️  Enrichment API failed: {e}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source, chunk.get("metadata", {}).get("doc_header", ""))
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            # Không có API key / API lỗi → fallback contextual prepend bằng tên tài liệu
            enriched_text = f"{context_line}\n\n{text}" if context_line else contextual_prepend(text, source)
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
