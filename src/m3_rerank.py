from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K, RERANK_MAX_LENGTH


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


_CROSS_ENCODER_MODEL = None


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        global _CROSS_ENCODER_MODEL
        if _CROSS_ENCODER_MODEL is None or getattr(_CROSS_ENCODER_MODEL, "_model_name_cached", None) != self.model_name:
            # Dùng sentence_transformers.CrossEncoder — KHÔNG dùng FlagEmbedding
            # (FlagReranker crash với transformers>=5.0 trên XLMRobertaTokenizer).
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER_MODEL = CrossEncoder(self.model_name)
            _CROSS_ENCODER_MODEL._model_name_cached = self.model_name
        self._model = _CROSS_ENCODER_MODEL
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        # max_length=512 (không phải 256): chunk 2048 ký tự tiếng Việt ~ 700-900 token,
        # cắt 256 làm mất đáp án nằm cuối đoạn → giảm context_recall đáng kể.
        scores = model.predict(pairs, max_length=RERANK_MAX_LENGTH, show_progress_bar=False)
        # predict() có thể trả về scalar khi chỉ 1 sample.
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(
            zip(scores, documents),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        results: list[RerankResult] = []
        for i, (score, doc) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            ))
        return results


class FlashrankReranker:
    """Lightweight ONNX alternative (<5ms) — chưa implement trong lab này.

    Raise NotImplementedError thay vì trả về [] : trả [] khiến caller tưởng
    rerank thành công rồi fallback nhầm sang kết quả hybrid chưa rerank.
    """
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # Tham khảo implement sau:
        # from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        raise NotImplementedError(
            "FlashrankReranker chưa được implement trong lab này — dùng CrossEncoderReranker."
        )


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")