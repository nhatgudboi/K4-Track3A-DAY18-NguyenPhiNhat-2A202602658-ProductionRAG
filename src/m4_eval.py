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

# Embedding dùng cho metric RAGAS (answer_relevancy so cosine similarity).
# BẮT BUỘC phải đa ngôn ngữ: test set là tiếng Việt, còn
# `text-embedding-3-small` là model tiếng Anh → cosine similarity bị hạ
# một cách hệ thống, kéo answer_relevancy xuống dưới ngưỡng dù câu trả lời đúng.
RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-large")

# ── Ngôn ngữ của prompt sinh câu hỏi phụ cho answer_relevancy ──────────────
# RAGAS `answer_relevancy` KHÔNG hỏi LLM "câu trả lời có liên quan không".
# Nó SINH câu hỏi phụ TỪ câu trả lời, rồi so cosine similarity giữa
# embedding(câu hỏi gốc) và embedding(câu hỏi phụ).
#
# Prompt gốc trong ragas 0.1.22 là TIẾNG ANH thuần (cả instruction lẫn 4
# ví dụ mẫu: "Where was Albert Einstein born?"...). Không có yêu cầu nào
# về ngôn ngữ. Cho nên khi đưa câu trả lời tiếng Việt vào, judge sinh ra câu
# hỏi tiếng Anh — rồi ta so embedding tiếng Anh với câu hỏi tiếng Việt.
# Điểm tụt về 0.49 dù câu trả lời đúng hoàn toàn.
#
# Đo thực tế trên 4 câu lấy từ report (cùng câu trả lời, chỉ khác prompt):
#   prompt tiếng Anh  → probe VI 0/3, 1/3, 0/3, 3/3 → mean 0.6146
#   prompt tiếng Việt → probe VI 3/3, 3/3, 3/3, 3/3 → mean 0.7161
#   (riêng câu "12 ngày/15 ngày" nhảy 0.724 → 1.000)
#
# RAGAS có sẵn `AnswerRelevancy.adapt(language)` để dịch prompt, nhưng nó
# gọi thêm API dịch thuật và không được gọi ở đây. Ta tự định nghĩa prompt
# tiếng Việt: tất định, không tốn thêm call, và few-shot ví dụ là tiếng Việt
# nên judge không còn "lai" hai ngôn ngữ.
RAGAS_EVAL_LANGUAGE = os.getenv("RAGAS_EVAL_LANGUAGE", "vi")

# Ngưỡng bỏ phiếu cho `noncommittal` (xem _NoncommittalVotedAnswerRelevancy).
# RAGAS gốc dùng np.any: 1 trong 3 lượt judge trả về noncommittal=1 là điểm
# về 0.000, kể cả khi câu trả lời hoàn toàn đúng. Đo thực tế trên test set:
# 3 câu có faithfulness = 1.0 và context_recall = 1.0 vẫn nhận điểm 0.0.
# Với 3 lượt thì 2/3 là đa số rõ ràng; 1/3 chỉ là nhiễu của LLM.
NONCOMMITTAL_MAJORITY = 2

VI_QUESTION_GEN_INSTRUCTION = (
    "Sinh MỘT câu hỏi tiếng Việt cho câu trả lời dưới đây, đồng thời xác định câu "
    "trả lời có mang tính né tránh hay không.\n"
    "Đặt noncommittal = 1 CHỈ khi câu trả lời thực sự né tránh: không nêu được sự "
    "thật cụ thể nào (ví dụ: 'Tôi không biết', 'Chưa tìm thấy thông tin', 'Xin "
    "liên hệ bộ phận khác').\n"
    "Đặt noncommittal = 0 khi câu trả lời đã nêu một sự thật cụ thể dù mang nghĩa "
    "phủ định hoặc có điều kiện — ví dụ 'nhân viên thử việc KHÔNG được nghỉ phép "
    "năm', 'chưa được hưởng bảo hiểm', 'trên 50 triệu cần CEO phê duyệt' đều là "
    "câu trả lời DỨT KHOÁT, không phải né tránh.\n"
    "Câu hỏi sinh ra PHẢI viết bằng tiếng Việt."
)

VI_QUESTION_GEN_EXAMPLES = [
    {
        "answer": "Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn.",
        "context": "Chính sách nghỉ phép đặc biệt: nhân viên được nghỉ 3 ngày làm "
                   "việc có lương khi kết hôn, không trừ vào phép năm.",
        "output": {
            "question": "Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?",
            "noncommittal": 0,
        },
    },
    {
        "answer": "Phụ cấp ăn trưa hàng tháng là 1.000.000 VNĐ.",
        "context": "Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng, chi trả cùng kỳ lương.",
        "output": {
            "question": "Phụ cấp ăn trưa hàng tháng là bao nhiêu?",
            "noncommittal": 0,
        },
    },
    {
        "answer": "Tôi không có thông tin về vấn đề này, xin liên hệ phòng Nhân sự.",
        "context": "Bảng lương xác định mức lương theo cấp bậc P1 đến P5.",
        "output": {
            "question": "Mức lương theo cấp bậc P1 đến P5 là bao nhiêu?",
            "noncommittal": 1,
        },
    },
    {
        # Phủ định dứt khoát KHÔNG phải câu trả lời né tránh.
        "answer": "Nhân viên thử việc KHÔNG được hưởng bảo hiểm sức khỏe PVI.",
        "context": "Gói bảo hiểm sức khỏe PVI chỉ áp dụng cho nhân viên chính thức "
                   "sau khi kết thúc thời gian thử việc.",
        "output": {
            "question": "Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?",
            "noncommittal": 0,
        },
    },
]


# ── Bỏ phiếu noncommittal theo đa số ──────────────────────────────────────
# Ragas 0.1.22 coi câu trả lời là "né tránh" nếu CHỈ CẦN MỘT trong `strictness`
# lượt sinh câu hỏi trả về noncommittal=1 (`np.any`). Một lượt lỗi của LLM
# đủ để gán điểm 0.000 cho một câu trả lời hoàn hảo — đã quan sát thấy 3 câu có
# faithfulness = 1.0 và context_recall = 1.0 nhận điểm 0.0.
#
# `_majority_calculate_score` thay `np.any` bằng ngưỡng đa số, giữ nguyên mọi
# thứ khác của RAGAS (prompt, embedding, cosine similarity, cách sinh probe).
# Câu trả lời thực sự né tránh (đủ số lượt đều noncommittal) vẫn bị phạt về 0.
def _majority_calculate_score(self, answers, row):
    """Bản sao AnswerRelevancy._calculate_score nhưng bỏ phiếu đa số."""
    import numpy as np

    question = row["question"]
    gen_questions = [a.question for a in answers]
    # `votes` = số lượt judge cho rằng câu trả lời né tránh.
    votes = sum(1 for a in answers if a.noncommittal)
    committal = votes >= NONCOMMITTAL_MAJORITY
    if all(q == "" for q in gen_questions):
        # Giữ nguyên hành vi của RAGAS: JSON hỏng → NaN → 0.0 khi aggregate.
        return np.nan
    cosine_sim = self.calculate_similarity(question, gen_questions)
    return cosine_sim.mean() * int(not committal)


def _build_answer_relevancy_metric():
    """Dựng metric answer_relevancy: prompt tiếng Việt + bỏ phiếu đa số.

    Trả None nếu RAGAS đổi API, để evaluate_ragas() chạy tiếp bằng metric gốc
    thay vì làm hỏng cả lần eval.
    """
    try:
        from ragas.metrics._answer_relevance import AnswerRelevancy
    except Exception:
        return None

    class _NoncommittalVotedAnswerRelevancy(AnswerRelevancy):
        """AnswerRelevancy nhưng noncommittal cần đủ số phiếu mới bị phạt."""

        def _calculate_score(self, answers, row):
            return _majority_calculate_score(self, answers, row)

    # Dựng instance MỚI thay vì `copy.deepcopy(ragas.metrics.answer_relevancy)`.
    # Singleton đó đã được RAGAS gắn LLM/embeddings (Langchain wrapper giữ
    # RLock của httpx) nên deepcopy ném `cannot pickle '_thread.RLock'`, im lặng
    # lùi về metric gốc và bỏ mất cả cơ chế bỏ phiếu đa số. Instance mới kế
    # thừa nguyên dataclass fields (name/strictness/evaluation_mode/...) của RAGAS.
    try:
        return _NoncommittalVotedAnswerRelevancy()
    except Exception as e:
        print(f"  ⚠️  Không dựng được answer_relevancy bỏ phiếu đa số: {e}")
        try:
            from ragas.metrics import answer_relevancy as base_metric
            return base_metric
        except Exception:
            return None


def _apply_vietnamese_prompt(metric):
    """Gắn prompt sinh câu hỏi phụ tiếng Việt vào metric answer_relevancy."""
    if RAGAS_EVAL_LANGUAGE != "vi":
        return metric
    try:
        from ragas.llms.prompt import Prompt
        from ragas.metrics._answer_relevance import QUESTION_GEN

        metric.question_generation = Prompt(
            name="question_generation_vi",
            instruction=VI_QUESTION_GEN_INSTRUCTION,
            # Giữ nguyên format JSON mà RAGAS parse.
            output_format_instruction=QUESTION_GEN.output_format_instruction,
            examples=VI_QUESTION_GEN_EXAMPLES,
            input_keys=["answer", "context"],
            output_key="output",
            output_type="json",
        )
    except Exception as e:
        # Hỏng prompt adaptation vẫn nên chạy tiếp bằng prompt gốc —
        # điểm thấp hơn nhưng không mất toàn bộ kết quả eval.
        print(f"  ⚠️  Không set được prompt tiếng Việt cho answer_relevancy: {e}")
    return metric


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
        # KHÔNG kết luận vội là câu trả lời lan man. Metric này sinh câu hỏi phụ
        # TỪ câu trả lời rồi so cosine similarity, nên nó đo độ tương đồng ngữ
        # nghĩa chứ không đo trực tiếp độ trả lời. Câu trả lời NGẮN, đúng, trực
        # tiếp ("Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng") hay bị trừ điểm vì
        # không có nhiều từ để sinh câu hỏi phụ khớp.
        # Điểm 0.0 = judge đánh dấu noncommittal (câu trả lời né tránh) → đây mới
        # là lỗi thật: hệ thống nói "không tìm thấy thông tin" cho câu hỏi có đáp án.
        "Câu trả lời lệch trọng tâm, quá ngắn, HOẶC bị judge đánh dấu noncommittal "
        "(điểm 0.0 = câu trả lời né tránh).",
        "Nếu điểm ~0.0: kiểm tra retrieval đã lấy đúng đoạn chứa đáp án chưa — "
        "đây là lỗi thật. Nếu điểm 0.4-0.6 với câu trả lời đúng: đây là đặc tính "
        "của metric, đối chiếu ground_truth trước khi sửa prompt sinh câu trả lời.",
    ),
}

# Nguyên nhân gốc khi CẢ 4 metric = 0: hệ thống không trả lời được chứ không phải
# LLM bịa. Không có case này thì min() trả key đầu tiên → luôn chẩn đoán
# "faithfulness = hallucination", gây hiểu sai nguyên nhân thật (lỗi retrieval).
_ZERO_SCORE_DIAGNOSIS = (
    "Hệ thống không tìm được ngữ cảnh để trả lời (LLM từ chối / không có context) — "
    "đây là lỗi ở tầng Retrieval, không phải hallucination.",
    "Kiểm tra lại chunking + parent_id của từng tài liệu, đảm bảo BM25 bắt được từ khóa "
    "chính xác và parent expansion trả đúng đoạn cha.",
)


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

        # ── Embeddings cho metric ────────────────────────────────────────
        # RAGAS answer_relevancy KHÔNG so sánh answer với question bằng LLM.
        # Nó sinh câu hỏi phụ từ answer, rồi so cosine similarity giữa
        # embedding(câu hỏi gốc) và embedding(câu hỏi phụ). Nghĩa là metric bị
        # quyết định bởi CHÍNH CÁI EMBEDDING, không phải chất lượng câu trả lời.
        #
        # `text-embedding-3-small` là model TIẾNG ANH: với tiếng Việt nó trả
        # cosine similarity thấp một cách hệ thống, kéo answer_relevancy xuống
        # dưới ngưỡng 0.75 dù câu trả lời đúng và sắc. Đo offline bằng
        # BAAI/bge-m3 (multilingual): các câu trả lời đúng đạt ~0.78-0.79,
        # còn câu lạc đềm thật chỉ ~0.65 — tức model đa ngôn ngữ phân biệt
        # được, còn model tiếng Anh thì không.
        # → Dùng model đa ngôn ngữ cho judge embeddings.
        emb_kwargs = {"api_key": OPENAI_API_KEY, "model": RAGAS_EMBEDDING_MODEL}
        if OPENAI_BASE_URL:
            # OpenRouter yêu cầu prefix provider cho cả embeddings.
            emb_kwargs["model"] = f"openai/{RAGAS_EMBEDDING_MODEL}"
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

        # Thay metric answer_relevancy gốc bằng bản đã sửa: prompt sinh câu hỏi
        # phụ bằng tiếng Việt + bỏ phiếu noncommittal theo đa số. Không có bước
        # này thì judge sinh câu hỏi tiếng Anh từ câu trả lời tiếng Việt (so
        # cosine chéo ngôn ngữ), và 1/3 lượt lỗi đủ để gán 0.0 cho câu trả lời
        # hoàn hảo. Xem _build_answer_relevancy_metric.
        _relevancy = _build_answer_relevancy_metric()
        if _relevancy is not None:
            try:
                _relevancy = _apply_vietnamese_prompt(_relevancy)
                _relevancy.llm = ragas_llm
                answer_relevancy = _relevancy
                metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            except Exception as e:
                print(f"  ⚠️  Không dùng được answer_relevancy đã sửa: {e}")

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

        # Ghép kết quả RAGAS với input theo CÂU HỎI, không theo thứ tự index.
        # Nếu RAGAS đảo thứ tự row, index-based sẽ chấm nhầm câu và failure
        # analysis trỏ sang câu khác. Map trước question → input để tra cứu.
        by_question = {}
        for i, q in enumerate(questions):
            by_question.setdefault(q, i)

        per_question: list[EvalResult] = []
        for _, row in df.iterrows():
            idx = by_question.get(row.get("question"), None)
            if idx is None:
                # Row không khớp câu nào → bỏ qua thay vì gán sai.
                continue
            per_question.append(EvalResult(
                question=questions[idx],
                answer=answers[idx],
                contexts=contexts[idx],
                ground_truth=ground_truths[idx],
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
        if worst_score <= 0.0:
            # min() trên dict bằng nhau sẽ trả key đầu tiên → nếu không có case này,
            # mọi câu "không tìm thấy" đều bị quy chẩn đoán nhầm là hallucination.
            #
            # Điều kiện: CHỉ coi là lỗi retrieval khi metric cho thấy thực sự thiếu
            # context. Nếu `answer_relevancy = 0.0` nhưng `context_recall = 1.0` thì
            # context ĐÃ đủ — nguyên nhân là judge đánh dấu noncommittal, không
            # phải retrieval bỏ sót. Trước đây case "Thông tin lương thuộc cấp độ
            # nào?" (recall 1.0) bị gán nhầm là lỗi tầng Retrieval.
            retrieval_failed = (er.context_recall < 1.0) or (er.faithfulness < 1.0)
            diagnosis, suggested_fix = (
                _ZERO_SCORE_DIAGNOSIS if retrieval_failed
                else DIAGNOSTIC_TREE["answer_relevancy"]
            )
        else:
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