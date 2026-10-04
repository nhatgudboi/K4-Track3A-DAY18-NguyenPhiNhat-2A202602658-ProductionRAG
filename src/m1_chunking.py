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
    """Lazy-load MiniLM để tránh load model nặng lúc import."""
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
    metadata = metadata or {}

    # 1) Tách câu — dùng regex gợi ý + filter bỏ câu rỗng.
    raw = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    sentences = [s.strip() for s in raw if s and s.strip()]
    if not sentences:
        return []

    # 2) Encode → vector.
    model = _get_semantic_model()
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    # 3) Cosine similarity liên tiếp (đã normalize → dot product).
    chunks: list[Chunk] = []
    current: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        # Vì đã normalize, cosine = dot product.
        sim = float(embeddings[i - 1] @ embeddings[i])
        if sim < threshold:
            # Chuyển ý → đóng chunk cũ, mở chunk mới.
            chunks.append(Chunk(
                text=" ".join(current).strip(),
                metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)},
            ))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(Chunk(
            text=" ".join(current).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)},
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _pack_paragraphs(paragraphs: list[str], max_size: int) -> list[str]:
    """Gộp paragraph liên tiếp cho tới khi gần max_size, rồi flush chunk."""
    groups: list[str] = []
    current = ""
    for p in paragraphs:
        # Nếu thêm p vào vượt max_size VÀ current đã có nội dung → flush.
        if current and len(current) + len(p) + 2 > max_size:
            groups.append(current.strip())
            current = ""
        current = (current + "\n\n" + p).strip() if current else p
    if current.strip():
        groups.append(current.strip())
    return [g for g in groups if g]


def _split_into_children(parent_text: str, child_size: int) -> list[str]:
    """Chia parent thành children: cố gắng bẻ tại ranh giới câu/cấu trúc."""
    children: list[str] = []
    buf = ""
    # Ưu tiên tách tại câu (. ? ! ) hoặc xuống dòng.
    parts = re.split(r'(?<=[.!?])\s+|\n+', parent_text)
    for p in parts:
        token = p.strip()
        if not token:
            continue
        if not buf:
            buf = token
            continue
        if len(buf) + len(token) + 1 > child_size:
            children.append(buf.strip())
            buf = token
        else:
            buf = f"{buf} {token}".strip()
    if buf.strip():
        children.append(buf.strip())
    # Nếu chunk đầu vẫn dài hơn child_size (đoạn không có dấu ngắt) → cắt cứng theo ký tự.
    final: list[str] = []
    for c in children:
        if len(c) <= child_size:
            final.append(c)
            continue
        for i in range(0, len(c), child_size):
            final.append(c[i:i + child_size])
    return final


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
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return [], []

    parent_groups = _pack_paragraphs(paragraphs, parent_size)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = metadata.get("source", "")
    prefix = f"{source}_" if source else ""

    for i, ptext in enumerate(parent_groups):
        pid = f"{prefix}parent_{i}"
        parents.append(Chunk(
            text=ptext,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": i},
        ))
        for ctext in _split_into_children(ptext, child_size):
            children.append(Chunk(
                text=ctext,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                parent_id=pid,
            ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}

    # Bắt header H1-H3 ở đầu dòng, kèm theo nội dung đi sau cho đến header kế tiếp.
    pattern = re.compile(r'^(#{1,3})\s+(.+?)\s*$', re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        # Không có header → trả về 1 chunk duy nhất.
        if text.strip():
            return [Chunk(text=text.strip(),
                          metadata={**metadata, "section": "(no header)", "strategy": "structure"})]
        return []

    chunks: list[Chunk] = []

    # Phần preamble (nội dung trước header đầu tiên).
    preamble = text[: matches[0].start()].strip()
    if preamble:
        chunks.append(Chunk(
            text=preamble,
            metadata={**metadata, "section": "(preamble)", "strategy": "structure"},
        ))

    # Duyệt qua từng header, gom nội dung tới header kế tiếp.
    for i, m in enumerate(matches):
        section_title = f"{m.group(1)} {m.group(2)}".strip()
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[content_start:content_end].strip()
        chunk_text = f"{section_title}\n\n{body}".strip() if body else section_title
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "section": section_title, "strategy": "structure", "chunk_index": len(chunks)},
        ))

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