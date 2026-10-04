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


# ─── Nhận diện chính sách bị thay thế ────────────────────────
# Kho dữ liệu có nhiều phiên bản của CÙNG một chính sách nằm song song
# (nghi_phep_nam_v2023.md vs nghi_phep_nam_v2024.md, mat_khau_v1.md vs
# mat_khau_v2.md). Nếu index tất cả, câu hỏi về chính sách hiện hành sẽ kéo
# cả đoạn của bản cũ lên → context_precision rớt và LLM trả lời lưỡng lự
# ("12 ngày theo 2023 và 15 ngày theo 2024").
#
# Mọi tài liệu .md đều có header dạng:
#   > Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự
# nên ta lấy metadata từ đó thay vì đoán theo tên file.

_VERSION_RE = re.compile(r"Phiên bản:\s*([\d.]+)", re.IGNORECASE)
_EFFECTIVE_RE = re.compile(r"Ngày hiệu lực:\s*(\d{2})/(\d{2})/(\d{4})", re.IGNORECASE)
_STATUS_RE = re.compile(r"Trạng thái:\s*([^|\n]+)", re.IGNORECASE)
_SUPERSEDED_BY_RE = re.compile(r"ĐÃ THAY THẾ bởi\s*v?([\d.]+)", re.IGNORECASE)
# Header H1: "# Chính sách nghỉ phép năm (Phiên bản 2023)"
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# Bỏ phần ngoặc chứa nhãn phiên bản ở cuối tiêu đề để gom về cùng một "họ".
_TITLE_SUFFIX_RE = re.compile(r"\s*\((?:[^)]*)\)\s*$")


def _title_base(title: str) -> str:
    """Bỏ nhãn phiên bản trong ngoặc ở cuối tiêu đề.

    "Chính sách nghỉ phép năm (Phiên bản 2023)" → "Chính sách nghỉ phép năm"
    "Chính sách mật khẩu (Phiên bản cũ)"       → "Chính sách mật khẩu"
    """
    return _TITLE_SUFFIX_RE.sub("", title).strip().lower()


def parse_policy_header(text: str) -> dict:
    """Trích metadata phiên bản từ header của tài liệu chính sách.

    Trả về dict với toàn bộ khoá (giá trị None nếu không tìm thấy) kể cả khi
    tài liệu không có header — đầu vào không đúng mẫu KHÔNG phải lỗi.
    """
    out: dict = {"policy_version": None, "effective_date": None,
                 "status": None, "superseded_by": None, "policy_family": None}

    m_ver = _VERSION_RE.search(text)
    if m_ver:
        out["policy_version"] = m_ver.group(1).strip()

    m_eff = _EFFECTIVE_RE.search(text)
    if m_eff:
        day, month, year = m_eff.groups()
        # Chuẩn hóa sang ISO để so sánh/sắp xếp được (và JSON-serializable).
        out["effective_date"] = f"{year}-{month}-{day}"

    m_st = _STATUS_RE.search(text)
    if m_st:
        out["status"] = m_st.group(1).strip()

    m_sup = _SUPERSEDED_BY_RE.search(text)
    if m_sup:
        out["superseded_by"] = m_sup.group(1).strip()

    m_h1 = _H1_RE.search(text)
    if m_h1:
        out["policy_family"] = _title_base(m_h1.group(1))

    return out


def mark_superseded_policies(docs: list[dict]) -> list[dict]:
    """Gắn cờ is_superseded cho các bản chính sách đã bị thay thế.

    Hai tín hiệu độc lập, đều đủ để kết luận:
      1. Header tự khai "ĐÃ THAY THẾ bởi vX" — tín hiệu mạnh, tài liệu tự nói.
      2. Cùng "họ chính sách" (tiêu đề bỏ nhãn phiên bản) nhưng ngày hiệu lực
         CŨ hơn bản mới nhất — suy ra từ dữ liệu, không phải tin tài liệu.

    Bản mới nhất trong mỗi họ là bản hiện hành và không bao giờ bị gắn cờ, kể cả
    khi chính nó tự khai đã bị thay thế — ngày hiệu lực là căn cứ mạnh hơn lời
    tuyên bố trong header. Tài liệu không có ngày hiệu lực (ví dụ PDF) không bị
    loại: không đủ căn cứ. Mọi doc đều có khoá is_superseded sau khi gọi hàm.
    """
    by_family: dict[str, list[dict]] = {}
    for doc in docs:
        family = doc.get("metadata", {}).get("policy_family")
        if family:
            by_family.setdefault(family, []).append(doc)

    # Mặc định mọi tài liệu là bản hiện hành. Gán False tường minh (không để
    # khoá thiếu) để caller dùng `meta["is_superseded"]` không bị KeyError và
    # để test khẳng định được "không phải bản cũ" thay vì "chưa biết".
    for doc in docs:
        doc["metadata"]["is_superseded"] = False

    for group in by_family.values():
        dated = [d for d in group if d["metadata"].get("effective_date")]
        if len(dated) < 2:
            # Chỉ có một bản → không đối chiếu được, không suy diễn là bản cũ.
            continue
        newest = max(dated, key=lambda d: d["metadata"]["effective_date"])
        newest_source = newest["metadata"].get("source", "")
        for doc in group:
            meta = doc["metadata"]
            # Bản mới nhất luôn là hiện hành, kể cả khi nó tự khai đã bị thay thế
            # (dữ liệu mâu thuẫn → tin dữ liệu, vì ngày hiệu lực là căn cứ mạnh).
            if doc is newest:
                continue
            if meta.get("superseded_by") or meta.get("effective_date"):
                meta["is_superseded"] = True
                meta["current_source"] = newest_source
    return docs


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR, drop_superseded: bool = False) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp, kèm metadata phiên bản lấy từ header tài liệu.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.

    Args:
        drop_superseded: nếu True, loại các bản chính sách đã bị thay thế
            (v2023 khi đã có v2024). Mặc định False → chỉ GẮN CỜ, không loại,
            để caller tự chọn giữa "loại hẳn" và "hạ thấp thứ hạng".
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        # Metadata phiên bản lấy từ header tài liệu, không suy đoán từ tên file.
        docs.append({"text": text,
                     "metadata": {"source": os.path.basename(fp),
                                  **parse_policy_header(text)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text,
                         "metadata": {"source": os.path.basename(fp),
                                      **parse_policy_header(text)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    # Gắn cờ bản bị thay thế SAU khi đã đọc hết, để so sánh được cả họ chính sách.
    mark_superseded_policies(docs)

    if drop_superseded:
        superseded = [d for d in docs if d["metadata"].get("is_superseded")]
        if superseded:
            names = ", ".join(d["metadata"]["source"] for d in superseded)
            print(f"  ℹ️  Đã loại {len(superseded)} bản chính sách bị thay thế: {names}")
        docs = [d for d in docs if not d["metadata"].get("is_superseded")]

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
    # Nếu chunk vẫn dài hơn child_size (đoạn không có dấu ngắt) → cắt cứng theo ký tự.
    # Cắt tại ranh giới từ (whitespace gần nhất phía sau) để không cắt giữa từ —
    # cắt giữa từ chính là lỗi mà module này sinh ra để tránh.
    final: list[str] = []
    for c in children:
        if len(c) <= child_size:
            final.append(c)
            continue
        buf = ""
        for token in c.split():
            if buf and len(buf) + len(token) + 1 > child_size:
                final.append(buf)
                buf = token
            else:
                buf = f"{buf} {token}".strip()
        if buf:
            final.append(buf)
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