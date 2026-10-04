from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TEST_SET_PATH, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
                    make_openai_client)

# RAGAS judge chỉ sinh output ngắn → giới hạn max_tokens để nhẹ tải API.
RAGAS_MAX_TOKENS = 512


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


# ─── Diagnostic Tree cho Failure Analysis ────────────────

DIAGNOSTIC_TREE = {
    "faithfulness": (
        "LLM tự bịa câu trả lời ngoài tài liệu (hallucination).",
        "Thắt chặt system prompt, giảm temperature về 0, ép LLM trích dẫn context trước khi trả lời.",
    ),
    "context_recall": (
        "Hệ thống tìm kiếm bỏ sót đoạn văn chứa đáp án đúng.",
        "Cải thiện chunking (hierarchical/structure-aware) hoặc bổ sung BM25 để bắt từ khóa chính xác.",
    ),
    "context_precision": (
        "Đoạn văn không liên quan bị xếp lên đầu, gây nhiễu LLM.",
        "Bổ sung Cross-Encoder reranking hoặc lọc theo metadata (source/section).",
    ),
    "answer_relevancy": (
        "Câu trả lời bị lệch trọng tâm câu hỏi, lan man.",
        "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp, yêu cầu chỉ trích xuất thông tin liên quan.",
    ),
}


def _to_float(value) -> float:
    """Chuyển score của RAGAS sang float; NaN/None → 0.0."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f  # NaN check


def _safe_mean(*values: float) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation với 4 metrics."""
    # Trả về zeros khi không có API key hoặc input rỗng.
    if not OPENAI_API_KEY or not questions:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }

    try:
        from ragas import evaluate
        from ragas.evaluation import RunConfig
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from datasets import Dataset

        # Cấu hình LLM + Embeddings dùng OpenRouter base_url.
        # max_tokens nhỏ: RAGAS chỉ cần output ngắn (verdict + lý do). Mặc định
        # ChatOpenAI để trần rất cao (16k) → OpenRouter từ chối với lỗi 402
        # "requested up to 16384 tokens but can only afford ..." khi balance thấp.
        chat_kwargs = {
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY,
            "max_tokens": RAGAS_MAX_TOKENS,
            "temperature": 0,
        }
        if OPENAI_BASE_URL:
            chat_kwargs["base_url"] = OPENAI_BASE_URL
        chat_llm = ChatOpenAI(**chat_kwargs)
        ragas_llm = LangchainLLMWrapper(chat_llm)

        emb_kwargs = {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"}
        if OPENAI_BASE_URL:
            # OpenRouter yêu cầu prefix provider cho cả embeddings.
            emb_kwargs["model"] = "openai/text-embedding-3-small"
            emb_kwargs["base_url"] = OPENAI_BASE_URL
        try:
            emb = OpenAIEmbeddings(**emb_kwargs)
        except Exception:
            emb = None

        # RAGAS 0.1.x mong đợi đúng tên cột: question / answer / contexts / ground_truth.
        rows = []
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            rows.append({
                "question": q,
                "answer": a,
                "contexts": [x for x in c if x] or ["(no context)"],
                "ground_truth": gt,
            })
        dataset = Dataset.from_list(rows)

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        # Wire LLM/embeddings cho từng metric cần thiết.
        for m in metrics:
            try:
                if hasattr(m, "llm"):
                    m.llm = ragas_llm
            except Exception:
                pass

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=emb,
            run_config=RunConfig(
                # OpenRouter giới hạn số request in-flight. Giảm max_workers +
                # tăng retry để tránh lỗi 402 in_flight_budget_exhausted làm mất score.
                max_workers=4,
                max_retries=5,
                max_wait=60,
                timeout=180,
            ),
            raise_exceptions=False,
        )
        df = result.to_pandas()

        per_question: list[EvalResult] = []
        for i, (_, row) in enumerate(df.iterrows()):
            per_question.append(EvalResult(
                question=questions[i],
                answer=answers[i],
                contexts=contexts[i],
                ground_truth=ground_truths[i],
                faithfulness=_to_float(row.get("faithfulness")),
                answer_relevancy=_to_float(row.get("answer_relevancy")),
                context_precision=_to_float(row.get("context_precision")),
                context_recall=_to_float(row.get("context_recall")),
            ))

        # Aggregate tính trên ĐÚNG toàn bộ câu hỏi.
        # pandas .mean() bỏ qua NaN → dễ thổi phồng điểm khi một vài câu lỗi API.
        # Ở đây NaN được coi là 0.0 (đúng với cột per_question) nên trung bình khớp.
        aggregate = {
            m: _safe_mean(*[getattr(er, m) for er in per_question])
            for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        }

        return {**aggregate, "per_question": per_question}

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    enriched = []
    for er in eval_results:
        metrics = {
            "faithfulness": er.faithfulness,
            "answer_relevancy": er.answer_relevancy,
            "context_precision": er.context_precision,
            "context_recall": er.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        avg = _safe_mean(*metrics.values())
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]
        enriched.append({
            "question": er.question,
            "answer": er.answer,
            "ground_truth": er.ground_truth,
            "worst_metric": worst_metric,
            "worst_score": float(worst_score),
            "avg_score": float(avg),
            "scores": {k: float(v) for k, v in metrics.items()},
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    enriched.sort(key=lambda x: x["avg_score"])
    return enriched[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")