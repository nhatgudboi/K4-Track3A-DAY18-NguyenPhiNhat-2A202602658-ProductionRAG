"""Tests for Module 1: Advanced Chunking."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m1_chunking import (chunk_basic, chunk_semantic, chunk_hierarchical,
                              chunk_structure_aware, compare_strategies, load_documents, Chunk,
                              parse_policy_header, mark_superseded_policies)

TEXT = """# Nghỉ phép

## Nghỉ phép năm

Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm.
Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên.

## Nghỉ phép không lương

Nhân viên có thể xin nghỉ phép không lương tối đa 30 ngày mỗi năm.
Đơn xin nghỉ phải được Giám đốc bộ phận phê duyệt.

## Nghỉ ốm

Cần nộp giấy xác nhận y tế trong vòng 3 ngày làm việc."""


# --- Baseline (đã implement sẵn) ---

def test_basic_returns_chunks():
    assert len(chunk_basic(TEXT)) > 0

def test_basic_type():
    assert all(isinstance(c, Chunk) for c in chunk_basic(TEXT))


# --- Semantic Chunking ---

def test_semantic_returns_chunks():
    result = chunk_semantic(TEXT, threshold=0.5)
    assert len(result) > 0, "Semantic chunking should return chunks"

def test_semantic_type():
    assert all(isinstance(c, Chunk) for c in chunk_semantic(TEXT, 0.5))

def test_semantic_groups_by_topic():
    """Semantic should produce fewer chunks than basic (groups related sentences)."""
    basic = chunk_basic(TEXT, chunk_size=100)
    semantic = chunk_semantic(TEXT, threshold=0.5)
    assert len(semantic) <= len(basic) + 2  # Allow some tolerance


# --- Hierarchical Chunking ---

def test_hierarchical_returns_both():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    assert len(parents) > 0, "Should return parents"
    assert len(children) > 0, "Should return children"

def test_hierarchical_children_have_parent_id():
    _, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    for c in children:
        assert c.parent_id is not None, "Each child must have parent_id"

def test_hierarchical_valid_parent_ids():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    parent_ids = {p.metadata.get("parent_id") for p in parents}
    for c in children:
        assert c.parent_id in parent_ids, f"Child parent_id '{c.parent_id}' not in parents"

def test_hierarchical_children_smaller():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    avg_p = sum(len(p.text) for p in parents) / max(len(parents), 1)
    avg_c = sum(len(c.text) for c in children) / max(len(children), 1)
    assert avg_c < avg_p, "Children should be smaller than parents"


# --- Structure-Aware Chunking ---

def test_structure_returns_chunks():
    result = chunk_structure_aware(TEXT)
    assert len(result) > 0, "Structure-aware should return chunks"

def test_structure_preserves_headers():
    result = chunk_structure_aware(TEXT)
    texts = " ".join(c.text for c in result)
    assert "Nghỉ phép năm" in texts, "Should preserve section headers"

def test_structure_has_section_metadata():
    result = chunk_structure_aware(TEXT)
    if result:
        assert any("section" in c.metadata for c in result), "Should have section in metadata"


# --- Compare ---

def test_compare_all_strategies():
    docs = load_documents()
    if docs:
        r = compare_strategies(docs)
        for key in ["basic", "semantic", "hierarchical", "structure"]:
            assert key in r, f"Missing strategy: {key}"


# --- Version-aware policy handling ---

V2023 = """# Chính sách nghỉ phép năm (Phiên bản 2023)
> Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2023 | Phòng ban: Nhân sự

## Số ngày phép năm
Mỗi nhân viên chính thức được hưởng **12 ngày phép năm** có lương."""

V2024 = """# Chính sách nghỉ phép năm (Phiên bản 2024)
> Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự

## Số ngày phép năm
Mỗi nhân viên chính thức được hưởng **15 ngày phép năm** có lương."""


def test_parse_policy_header():
    meta = parse_policy_header(V2024)
    assert meta["policy_version"] == "2.0"
    assert meta["effective_date"] == "2024-01-01"
    # Nhãn "(Phiên bản 2024)" phải được bỏ để gom về cùng họ chính sách.
    assert meta["policy_family"] == "chính sách nghỉ phép năm"


def test_parse_policy_header_handles_missing():
    """Tài liệu không có header không phải lỗi — trả về None, không raise."""
    meta = parse_policy_header("Một đoạn văn bất kỳ không có header.")
    assert meta["policy_version"] is None
    assert meta["effective_date"] is None
    assert meta["policy_family"] is None


def test_parse_policy_header_detects_superseded():
    old = """# Chính sách mật khẩu (Phiên bản cũ)
> Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2022 | Trạng thái: ĐÃ THAY THẾ bởi v2.0
"""
    assert parse_policy_header(old)["superseded_by"] == "2.0"


def test_superseded_detected_by_effective_date():
    """Bản cũ hơn trong cùng họ bị gắn cờ dù KHÔNG tự khai đã thay thế."""
    docs = [
        {"text": V2023, "metadata": {**parse_policy_header(V2023), "source": "old.md"}},
        {"text": V2024, "metadata": {**parse_policy_header(V2024), "source": "new.md"}},
    ]
    mark_superseded_policies(docs)
    assert docs[0]["metadata"]["is_superseded"] is True
    assert docs[0]["metadata"]["current_source"] == "new.md"
    assert docs[1]["metadata"]["is_superseded"] is False


def test_single_version_not_marked_superseded():
    """Chỉ có một bản thì không đối chiếu được → không suy diễn là bản cũ."""
    docs = [{"text": V2024, "metadata": {**parse_policy_header(V2024), "source": "only.md"}}]
    mark_superseded_policies(docs)
    assert docs[0]["metadata"].get("is_superseded") is False


def test_load_documents_flags_real_superseded():
    """Kho dữ liệu thật phải có đúng 2 bản bị thay thế."""
    docs = load_documents()
    by_source = {d["metadata"]["source"]: d["metadata"] for d in docs}
    assert by_source["nghi_phep_nam_v2023.md"]["is_superseded"] is True
    assert by_source["mat_khau_v1.md"]["is_superseded"] is True
    # Bản hiện hành KHÔNG được gắn cờ.
    assert by_source["nghi_phep_nam_v2024.md"]["is_superseded"] is False
    assert by_source["mat_khau_v2.md"]["is_superseded"] is False
    # Tài liệu không thuộc họ nào (chỉ có một bản) vẫn phải an toàn.
    assert by_source["tam_ung.md"]["is_superseded"] is False


def test_load_documents_can_drop_superseded():
    kept = load_documents(drop_superseded=True)
    assert len(kept) < len(load_documents())
    assert not any(d["metadata"].get("is_superseded") for d in kept)
