from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K, RRF_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"
    chunk_id: str | None = None  # định danh ổn định để RRF dedup không nhầm chunk trùng text


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        # underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép") → thay bằng space
        # để BM25 tokenize theo split(" ") khớp với query.
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception as e:
        # Fallback an toàn nếu underthesea lỗi (network/model missing): lower + bỏ punctuation.
        import re
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            seg = segment_vietnamese(chunk["text"])
            tokens = seg.split() if seg else []
            self.corpus_tokens.append(tokens)
        from rank_bm25 import BM25Okapi
        # `any(self.corpus_tokens)` sai: any() trên list-of-list chỉ False khi MỌI
        # phần tử rỗng → corpus toàn rỗng vẫn khởi tạo BM25Okapi rồi crash.
        if self.corpus_tokens and any(tokens for tokens in self.corpus_tokens):
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
        seg = segment_vietnamese(query)
        tokenized_query = seg.split() if seg else []
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        # Lấy top_k ứng viên, lọc score > 0.
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        results: list[SearchResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[idx]),
                metadata=doc.get("metadata", {}),
                method="bm25",
                chunk_id=str(doc.get("chunk_id", idx)),
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.using_fallback = False
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
            self.client.get_collections()
        except Exception as e:
            # Fallback in-memory để test vẫn chạy được khi thiếu Docker.
            # Phải CẢNH BÁO rõ — im lặng fallback khiến người dùng tưởng đang
            # chạy với Qdrant thật và tưởng điểm số không đáng tin.
            self.client = QdrantClient(":memory:")
            self.using_fallback = True
            print(f"  ⚠️  Không kết nối được Qdrant server ({QDRANT_HOST}:{QDRANT_PORT}): {e}")
            print("  ⚠️  Đang dùng Qdrant in-memory fallback. Hãy chạy 'docker compose up -d' "
                  "để tái tạo đúng môi trường Production.")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return
        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Recreate collection để tránh conflict schema khi chạy lại.
        try:
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        except Exception:
            # In-memory client không có recreate; bỏ qua.
            pass

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True, normalize_embeddings=True)
        points = [
            PointStruct(id=i, vector=v.tolist(),
                        payload={**c.get("metadata", {}), "text": c["text"]})
            for i, (v, c) in enumerate(zip(vectors, chunks))
        ]
        # wait=True đảm bảo points đã được commit trước khi query.
        self.client.upsert(collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        try:
            query_vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        except Exception:
            return []
        try:
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except TypeError:
            # Tương thích phiên bản cũ — phải giữ with_payload=True, nếu không
            # payload=None → text rỗng → dense search trả về rác.
            response = self.client.query_points(
                collection, query=query_vector, limit=top_k, with_payload=True
            )
        results: list[SearchResult] = []
        for pt in response.points:
            payload = pt.payload or {}
            results.append(SearchResult(
                text=payload.get("text", ""),
                score=float(pt.score) if pt.score is not None else 0.0,
                metadata={k: v for k, v in payload.items() if k != "text"},
                method="dense",
                chunk_id=str(pt.id) if pt.id is not None else None,
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = RRF_K,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """
    Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1).

    Dedup theo `chunk_id` (định danh ổn định) thay vì theo `text`: nếu khóa theo
    text, hai chunk trùng nội dung sẽ bị gộp làm một và chỉ giữ metadata của
    chunk đầu tiên — cùng dạng lỗi "parent-child collision" ở M1.
    """
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.chunk_id or f"text::{result.text}"
            if key not in rrf_scores:
                # Lưu reference đến SearchResult đầu tiên (giữ metadata gốc).
                rrf_scores[key] = {"score": 0.0, "result": result}
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
            chunk_id=item["result"].chunk_id,
        )
        for item in sorted_items
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()
        # True nếu đang dùng Qdrant thật (không phải fallback in-memory).
        self.using_qdrant_server = getattr(self.dense.client, "_client", None) is not None

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")