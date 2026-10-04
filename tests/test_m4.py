"""Tests for Module 4: Evaluation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, EvalResult

def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics():
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]


# --- Phân biệt lỗi Retrieval với lỗi chấm điểm ---

def test_zero_relevancy_with_perfect_context_not_blamed_on_retrieval():
    """context_recall = 1.0 + answer_relevancy = 0.0 → KHÔNG phải lỗi retrieval.

    Judge đánh dấu noncommittal thì ar = 0.0 dù context đã đầy đủ. Trước đây
    case này bị gán nhầm là "Hệ thống không tìm được ngữ cảnh".
    """
    results = [EvalResult("Q1", "A1", ["C1"], "GT1",
                          1.0, 0.0, 1.0, 1.0)]
    f = failure_analysis(results, bottom_n=1)[0]
    assert "tầng Retrieval" not in f["diagnosis"], \
        "Context đã đủ (recall=1.0) nên không được chẩn đoán lỗi retrieval"
    assert f["worst_metric"] == "answer_relevancy"


def test_zero_with_missing_context_blamed_on_retrieval():
    """faithfulness = 0.0 + context_recall thấp → đúng là lỗi retrieval."""
    results = [EvalResult("Q1", "A1", ["C1"], "GT1",
                          0.0, 0.0, 0.0, 0.0)]
    f = failure_analysis(results, bottom_n=1)[0]
    assert "Retrieval" in f["diagnosis"]


def test_bottom_n_sorts_by_avg_ascending():
    """Failures phải được sắp xếp tăng dần theo điểm trung bình."""
    results = [
        EvalResult("good", "a", ["c"], "g", 1.0, 1.0, 1.0, 1.0),
        EvalResult("bad", "a", ["c"], "g", 0.0, 0.1, 0.1, 0.1),
        EvalResult("mid", "a", ["c"], "g", 0.5, 0.5, 0.5, 0.5),
    ]
    f = failure_analysis(results, bottom_n=3)
    assert [x["question"] for x in f] == ["bad", "mid", "good"]
