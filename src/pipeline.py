from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY, OPENAI_MODEL, make_openai_client


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    # parent_id → parent text, dùng để "expand" child về parent trước khi đưa cho LLM.
    parent_lookup: dict[str, str] = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for parent in parents:
            parent_lookup[parent.metadata.get("parent_id")] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    cache_path = os.path.join("reports", "_cache", "enriched_chunks.json")
    if os.path.exists(cache_path):
        import json as _json
        try:
            with open(cache_path, encoding="utf-8") as f:
                all_chunks = _json.load(f)
            print(f"  ✓ Loaded {len(all_chunks)} enriched chunks from cache ({time.time()-t0:.1f}s)", flush=True)
        except Exception:
            all_chunks = []
    if not all_chunks or not os.path.exists(cache_path):
        enriched = enrich_chunks(all_chunks)
        if enriched:
            # Giữ cả text đã làm giàu (để embed) và parent_id (để expand về parent khi sinh answer).
            all_chunks = [
                {"text": e.enriched_text,
                 "metadata": {**e.auto_metadata, "parent_id": e.auto_metadata.get("parent_id")}}
                for e in enriched
            ]
            import json as _json
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                _json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
        else:
            print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker, parent_lookup


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              parent_lookup: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)

    # Parent expansion: chunk nào trỏ tới parent_id thì trả về parent text cho LLM.
    # Mục tiêu: tăng context_recall (parent chứa đầy đủ bối cảnh so với child 256 chars).
    seen_parents: set[str] = set()
    contexts: list[str] = []
    for r in (reranked if reranked else results[:3]):
        text = r.text
        if parent_lookup:
            pid = r.metadata.get("parent_id")
            if pid and pid in parent_lookup and pid not in seen_parents:
                seen_parents.add(pid)
                text = parent_lookup[pid]
        if text not in contexts:
            contexts.append(text)

    from config import OPENAI_API_KEY, OPENAI_MODEL, make_openai_client
    if OPENAI_API_KEY and contexts:
        try:
            client = make_openai_client()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "Bạn là trợ lý nội bộ. Trả lời CHỈ dựa trên context được cung cấp. "
                        "Nếu context không chứa thông tin cần thiết, nói rõ 'Không tìm thấy.'. "
                        "Trích dẫn câu ngắn từ context khi có thể. Trả lời bằng tiếng Việt."
                    )},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
                max_tokens=500,
                temperature=0,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker,
                      parent_lookup: dict[str, str] | None = None):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, parent_lookup)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, parent_lookup = build_pipeline()
    evaluate_pipeline(search, reranker, parent_lookup)
    print(f"\nTotal: {time.time() - start:.1f}s")
