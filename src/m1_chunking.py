from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


_SEMANTIC_MODEL = None


def _get_semantic_model():
    """Lazy-load + cache sentence encoder (tránh load lại mỗi lần gọi)."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s and s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    embeddings = _get_semantic_model().encode(sentences, show_progress_bar=False)

    def cosine_sim(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    groups, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append(current)
            current = []
        current.append(sentences[i])
    groups.append(current)

    return [Chunk(text=" ".join(g), metadata={**metadata, "chunk_index": i, "strategy": "semantic"})
            for i, g in enumerate(groups)]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _split_to_size(text: str, max_size: int) -> list[str]:
    """Chia 1 đoạn quá dài thành các mảnh ≤ max_size: ưu tiên ranh giới câu, rồi đến từ."""
    if len(text) <= max_size:
        return [text]
    units = [s for s in re.split(r'(?<=[.!?])\s+|\n', text) if s.strip()]
    pieces = []
    for unit in units:
        if len(unit) <= max_size:
            pieces.append(unit)
            continue
        words, cur = unit.split(), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_size:
                pieces.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        if cur:
            pieces.append(cur)
    return pieces


def _pack(units: list[str], max_size: int, sep: str) -> list[str]:
    """Gộp tham lam các unit liên tiếp sao cho mỗi nhóm ≤ max_size."""
    packed, cur = [], ""
    for unit in units:
        for piece in _split_to_size(unit, max_size):
            if cur and len(cur) + len(sep) + len(piece) > max_size:
                packed.append(cur)
                cur = piece
            else:
                cur = f"{cur}{sep}{piece}" if cur else piece
    if cur:
        packed.append(cur)
    return packed


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Prefix theo source để parent_id duy nhất khi index nhiều document chung 1 collection
    prefix = f"{metadata['source']}::" if metadata.get("source") else ""

    parents, children = [], []
    for parent_text in _pack(paragraphs, parent_size, "\n\n"):
        pid = f"{prefix}parent_{len(parents)}"
        parents.append(Chunk(text=parent_text,
                             metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
        child_units = [p.strip() for p in parent_text.split("\n\n") if p.strip()]
        for child_text in _pack(child_units, child_size, "\n\n"):
            children.append(Chunk(text=child_text,
                                  metadata={**metadata, "chunk_type": "child",
                                            "child_index": len(children)},
                                  parent_id=pid))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    header, content = "", ""

    def _flush():
        body = content.strip()
        if not body:
            # Header không có nội dung (VD: title H1 ngay trước H2) → không tạo chunk rỗng
            return
        chunk_text = f"{header}\n{body}" if header else body
        chunks.append(Chunk(text=chunk_text, metadata={
            **metadata,
            "section": header.lstrip("#").strip() if header else "",
            "chunk_index": len(chunks),
            "strategy": "structure",
        }))

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            _flush()
            header, content = part.strip(), ""
        else:
            content += part
    _flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
