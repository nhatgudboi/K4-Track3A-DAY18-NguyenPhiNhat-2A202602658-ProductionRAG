"""Tests for Module 3: Reranking."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m3_rerank import CrossEncoderReranker, benchmark_reranker, RerankResult

Q = "Nhân viên được nghỉ phép bao nhiêu ngày?"
DOCS = [
    {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
    {"text": "VPN dùng WireGuard AES-256.", "score": 0.6, "metadata": {}},
]

def test_rerank_returns():
    r = CrossEncoderReranker().rerank(Q, DOCS, top_k=2)
    assert len(r) > 0 and len(r) <= 2

def test_rerank_type():
    assert all(isinstance(x, RerankResult) for x in CrossEncoderReranker().rerank(Q, DOCS))

def test_rerank_sorted():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if len(r) >= 2:
        assert r[0].rerank_score >= r[1].rerank_score

def test_rerank_relevant_first():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if r:
        assert "nghỉ" in r[0].text.lower() or "12" in r[0].text

def test_benchmark_stats():
    stats = benchmark_reranker(CrossEncoderReranker(), Q, DOCS, n_runs=2)
    assert "avg_ms" in stats and "min_ms" in stats and "max_ms" in stats


# --- Sắp xếp lại ứng viên theo phiên bản chính sách ---

def test_demote_moves_superseded_to_end():
    from src.pipeline import _demote_superseded_candidates
    from src.m3_rerank import RerankResult

    ranked = [
        RerankResult("v2023: 12 ngày", 0.9, 9.0, {"is_superseded": True}, 0),
        RerankResult("v2024: 15 ngày", 0.9, 9.1, {"is_superseded": False}, 1),
        RerankResult("v2023: thêm 1 ngày", 0.8, 8.0, {"is_superseded": True}, 2),
    ]
    out = _demote_superseded_candidates(ranked, [])
    assert out[0].text == "v2024: 15 ngày", "Bản hiện hành phải lên đầu"
    # Thứ tự giữa các bản cũ được giữ nguyên (ổn định).
    assert [r.text for r in out[1:]] == ["v2023: 12 ngày", "v2023: thêm 1 ngày"]


def test_demote_keeps_all_items():
    """Không được mất chunk nào — bản cũ vẫn cần cho câu hỏi về lịch sử."""
    from src.pipeline import _demote_superseded_candidates
    from src.m3_rerank import RerankResult

    ranked = [RerankResult(f"d{i}", 0.5, 1.0, {"is_superseded": i % 2 == 0}, i)
              for i in range(6)]
    out = _demote_superseded_candidates(ranked, [])
    assert len(out) == 6
    assert sorted(r.text for r in out) == sorted(r.text for r in ranked)


def test_demote_handles_empty():
    from src.pipeline import _demote_superseded_candidates
    assert _demote_superseded_candidates([], []) == []


def test_demote_without_metadata_key():
    """Chunk không có khoá is_superseded phải được coi là bản hiện hành."""
    from src.pipeline import _demote_superseded_candidates
    from src.m3_rerank import RerankResult

    ranked = [RerankResult("a", 0.5, 1.0, {}, 0), RerankResult("b", 0.4, 1.0, {}, 1)]
    out = _demote_superseded_candidates(ranked, [])
    assert [r.text for r in out] == ["a", "b"]


def test_rerank_pool_larger_than_top_k():
    """Pool rerank phải rộng hơn RERANK_TOP_K, nếu không việc hạ bản cũ vô nghĩa."""
    from src.pipeline import RERANK_POOL_SIZE
    from config import RERANK_TOP_K
    assert RERANK_POOL_SIZE > RERANK_TOP_K
