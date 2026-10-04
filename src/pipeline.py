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
from src.m3_rerank import CrossEncoderReranker, benchmark_reranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import (RERANK_TOP_K, HYBRID_TOP_K, HIERARCHICAL_PARENT_SIZE,
                    HIERARCHICAL_CHILD_SIZE, OPENAI_API_KEY, OPENAI_MODEL, make_openai_client)

# Latency tích luỹ từng bước (ms) — in ra cuối run để đối chiếu SLA Production.
LATENCY: dict[str, float] = {}

# Phiên bản schema cache enrichment. Tăng khi metadata chunk đổi hình dạng,
# để cache cũ bị loại thay vì tái dùng với metadata thiếu khoá mới.
# v2: thêm is_superseded / effective_date / policy_family.
PIPELINE_CACHE_VERSION = 2

# Số ứng viên đưa qua cross-encoder TRƯỚC khi lọc bản chính sách cũ.
# Phải > RERANK_TOP_K, nếu không bản cũ chiếm hết suất và việc hạ thấp
# thứ hạng trở nên vô nghĩa. Số này không ảnh hưởng latency quá nhiều vì
# rerank chạy theo batch.
RERANK_POOL_SIZE = 8

# System prompt sinh câu trả lời. Ba quy tắc bổ sung dưới đây sửa các lỗi đo
# được, không phải để "nghe hay hơn":
#
# 1. TÍNH PRO-RATA TRÊN ĐƠN VỊ THÁNG. Lỗi thật đã quan sát: câu hỏi "tạm ứng 15
#    triệu, trả sau 20 ngày, phạt bao nhiêu?" bị trả lời 300.000 VNĐ — tức áp
#    đúng 2%/tháng cho 5 ngày quá hạn mà không nhân theo số ngày. Đúng là
#    15.000.000 × 2% × 5/30 = 50.000 VNĐ. RAGAS chấm faithfulness 0.167.
# 2. ƯU TIÊN CHÍNH SÁCH HIỆN HÀNH. Kho có cả v2023 và v2024 cùng chủ đề; nếu
#    cả hai lọt vào context, LLM đáp lưỡng lự "12 ngày và 15 ngày".
# 3. TÍNH TOÁN PHẢI CHỦ ĐỘNG, không bỏ trống. Câu hỏi suy diễn cần LLM tự
#    tính; yêu cầu hiển thị từng bước để lỗi số học lộ ra (RAGAS bắt được).
_GENERATION_SYSTEM_PROMPT = (
    "Bạn là trợ lý nội bộ. Trả lời CHỈ dựa trên context được cung cấp. "
    "Nếu context không chứa thông tin cần thiết, nói rõ 'Không tìm thấy.'. "
    "Trích dẫn câu ngắn từ context khi có thể. Trả lời bằng tiếng Việt.\n\n"
    "Quy tắc bắt buộc:\n"
    "1. Nếu context có nhiều phiên bản của cùng một chính sách, chỉ dùng phiên bản "
    "ĐANG HIỆN HÀNH (ngày hiệu lực mới nhất / bản được ghi là thay thế bản cũ). "
    "Không liệt kê hay so sánh các bản cũ.\n"
    "2. Khi cần tính toán, PHẢI quy đổi về cùng đơn vị trước khi nhân. Ví dụ phí "
    "ghi theo tháng nhưng áp dụng cho N ngày thì: tiền × tỉ lệ × (N ÷ 30). "
    "Nêu công thức và từng bước tính. Không được dùng tỉ lệ nguyên cho số ngày lẻ.\n"
    "3. Nêu con số cuối cùng một lần, rõ ràng."
)


def _chunk_fingerprint(n_chunks: int) -> str:
    """Fingerprint của lần chunking, dùng để validate cache enrichment.

    Nếu không có fingerprint, cache cũ vẫn được dùng sau khi đổi
    parent_size/child_size → `pid in parent_lookup` False → parent expansion
    biến mất IM LẶNG, chất lượng context tụt mà không có lỗi nào báo.

    `PIPELINE_CACHE_VERSION` nằm trong fingerprint để cache cũ bị loại khi
    schema metadata thay đổi (ví dụ thêm cờ is_superseded). Nếu không, cache
    cũ vẫn "hợp lệ" về số chunk nhưng thiếu khoá mới → bộ lọc chính sách im
    lặng không chạy, đúng loại lỗi im lặng mà fingerprint sinh ra để chống.
    """
    import hashlib
    raw = (f"{PIPELINE_CACHE_VERSION}|{n_chunks}|{HIERARCHICAL_PARENT_SIZE}"
           f"|{HIERARCHICAL_CHILD_SIZE}")
    return hashlib.sha256(raw.encode()).hexdigest()[:12]



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
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
                # ID ổn định để RRF dedup không nhầm hai chunk trùng nội dung.
                "chunk_id": f"{doc['metadata'].get('source','')}#{child.parent_id}#{len(all_chunks)}",
            })
    LATENCY["1_chunking_ms"] = (time.time() - t0) * 1000
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)
    if len(parent_lookup) < len(docs):
        print(f"  ⚠️  parent_id KHÔNG duy nhất: {len(parent_lookup)}/{len(docs)} docs", flush=True)

    # Chunk từ chính sách đã bị thay thế vẫn được index (không xóa — xóa là mất
    # khả năng trả lời câu hỏi về lịch sử chính sách). Cờ is_superseded đã nằm
    # trong metadata của chunk; việc hạ thấp thứ hạng thực hiện lúc truy vấn,
    # sau rerank — xem _demote_superseded_candidates.
    n_stale = sum(1 for c in all_chunks if c["metadata"].get("is_superseded"))
    if n_stale:
        print(f"  ℹ️  {n_stale} chunks thuộc chính sách đã bị thay thế "
              f"(sẽ bị hạ thấp khi truy vấn)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    cache_path = os.path.join("reports", "_cache", "enriched_chunks.json")
    fingerprint = _chunk_fingerprint(len(all_chunks))
    raw_chunks = all_chunks
    if os.path.exists(cache_path):
        import json as _json
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = _json.load(f)
            # Cache chỉ hợp lệ khi fingerprint khớp VÀ đủ số chunk.
            if (isinstance(cached, dict)
                    and cached.get("_fingerprint") == fingerprint
                    and len(cached.get("chunks", [])) == len(all_chunks)):
                all_chunks = cached["chunks"]
                print(f"  ✓ Loaded {len(all_chunks)} enriched chunks from cache ({time.time()-t0:.1f}s)", flush=True)
            else:
                print("  ⚠️  Cache enrichment không khớp fingerprint (chunking đã đổi) → làm giàu lại", flush=True)
                all_chunks = []
        except Exception:
            all_chunks = []
    if not all_chunks or not os.path.exists(cache_path):
        enriched = enrich_chunks(raw_chunks)
        if enriched:
            # Giữ cả text đã làm giàu (để embed) và parent_id (để expand về parent khi sinh answer).
            all_chunks = [
                {"text": e.enriched_text,
                 "metadata": {**e.auto_metadata, "parent_id": e.auto_metadata.get("parent_id")},
                 "chunk_id": raw_chunks[i].get("chunk_id")}
                for i, e in enumerate(enriched)
            ]
            import json as _json
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                _json.dump({"_fingerprint": fingerprint, "chunks": all_chunks},
                           f, ensure_ascii=False, indent=2)
            print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
        else:
            print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    LATENCY["2_enrichment_ms"] = (time.time() - t0) * 1000

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    LATENCY["3_indexing_ms"] = (time.time() - t0) * 1000
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    # Warm-up + đo latency thật của từng bước truy vấn (M2 / M3).
    _warmup_and_measure(search, reranker)

    return search, reranker, parent_lookup


def _warmup_and_measure(search: HybridSearch, reranker: CrossEncoderReranker) -> None:
    """Đo latency per-query của hybrid search + rerank (trừ LLM generation).

    Đây là latency tầng retrieval của Production RAG — con số cần theo dõi để
    đảm bảo đáp ứng SLA (<150ms cho rerank theo khuyến nghị của lab).
    """
    probe = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    try:
        results = search.search(probe)
        docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
        if not docs:
            return
        t = time.perf_counter()
        for _ in range(3):
            search.search(probe)
        LATENCY["4a_hybrid_search_ms"] = (time.perf_counter() - t) / 3 * 1000

        stats = benchmark_reranker(reranker, probe, docs, n_runs=3)
        LATENCY["4b_rerank_ms"] = stats["avg_ms"]
    except Exception as e:
        print(f"  ⚠️  Không đo được latency: {e}", flush=True)


def print_latency_report() -> None:
    """In bảng latency từng bước: offline (build) + per-query (runtime)."""
    offline = [
        ("1_chunking_ms", "1. Chunking (offline, toàn kho)"),
        ("2_enrichment_ms", "2. Enrichment M5 (offline, 1 call/chunk)"),
        ("3_indexing_ms", "3. Indexing BM25 + Dense (offline)"),
    ]
    online = [
        ("q_hybrid_search_ms", "4a. Hybrid Search BM25+Dense+RRF"),
        ("q_rerank_ms", "4b. Cross-Encoder Rerank top-20→3"),
        ("q_llm_generation_ms", "5. LLM answer generation"),
        ("q_end_to_end_ms", "→ END-TO-END per query"),
    ]
    print("\n" + "=" * 62)
    print("LATENCY BREAKDOWN")
    print("=" * 62)
    print("  [Offline — chạy 1 lần khi build index]")
    for key, label in offline:
        if key in LATENCY:
            print(f"    {label:<44} {LATENCY[key]:>10.1f} ms")
    print("  [Online — trung bình trên test set]")
    for key, label in online:
        if key in LATENCY:
            print(f"    {label:<44} {LATENCY[key]:>10.1f} ms")

    # SLA: lab khuyến nghị giữ rerank < 150ms để người dùng không phải chờ.
    if "q_rerank_ms" in LATENCY:
        ok = LATENCY["q_rerank_ms"] < 150
        print(f"  {'✓' if ok else '⚠️ '} SLA rerank < 150ms: "
              f"{'PASS' if ok else 'FAIL'} ({LATENCY['q_rerank_ms']:.1f} ms)")


def _demote_superseded_candidates(reranked, results):
    """Đẩy chunk từ chính sách bị thay thế xuống cuối, thay vì xóa hẳn.

    Vì sao không xóa: câu hỏi hợp lệ về lịch sử chính sách ("chính sách cũ quy
    định mấy ngày phép?") vẫn cần bản cũ. Xóa sẽ trả lời sai kiểu "không tìm
    thấy". Vì sao phải hạ: câu hỏi về chính sách HIỆN HÀNH mà để bản cũ chen
    vào top-3 sẽ khiến LLM đáp lưỡng lự ("12 ngày và 15 ngày") — đây là nguyên
    nhân context_precision rớt về 0.333 ở các câu hỏi về phép năm.

    Áp dụng SAU rerank, vì cross-encoder không biết ngày hiệu lực — nó chỉ so
    độ liên quan về mặt ngữ nghĩa, mà "nghỉ 12 ngày" và "nghỉ 15 ngày" về mặt
    ngữ nghĩa gần như giống nhau.
    """
    if not reranked:
        return reranked
    current = [r for r in reranked if not r.metadata.get("is_superseded")]
    stale = [r for r in reranked if r.metadata.get("is_superseded")]
    return current + stale


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              parent_lookup: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t = time.perf_counter()
    results = search.search(query)
    LATENCY["_t_search"] = LATENCY.get("_t_search", 0.0) + (time.perf_counter() - t) * 1000

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata,
             "chunk_id": r.chunk_id} for r in results]
    t = time.perf_counter()
    # Rerank trên pool RỘNG HƠN RERANK_TOP_K, rồi mới lọc bản cũ và cắt còn
    # RERANK_TOP_K. Nếu cắt top-3 ngay trong rerank() thì bản chính sách cũ đã
    # chiếm hết 3 suất và _demote_superseded_candidates không còn gì để dời.
    reranked = reranker.rerank(query, docs, top_k=RERANK_POOL_SIZE)
    LATENCY["_t_rerank"] = LATENCY.get("_t_rerank", 0.0) + (time.perf_counter() - t) * 1000
    LATENCY["_q_count"] = LATENCY.get("_q_count", 0) + 1

    # Bản chính sách bị thay thế đi sau bản hiện hành (xem _demote_superseded_candidates).
    reranked = _demote_superseded_candidates(reranked, results)[:RERANK_TOP_K]

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

    t_gen = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            client = make_openai_client()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
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
    LATENCY["_t_gen"] = LATENCY.get("_t_gen", 0.0) + (time.perf_counter() - t_gen) * 1000
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

    # Trung bình latency thật trên test set.
    n = LATENCY.get("_q_count", 0)
    if n:
        LATENCY["q_hybrid_search_ms"] = LATENCY.get("_t_search", 0.0) / n
        LATENCY["q_rerank_ms"] = LATENCY.get("_t_rerank", 0.0) / n
        LATENCY["q_llm_generation_ms"] = LATENCY.get("_t_gen", 0.0) / n
        LATENCY["q_end_to_end_ms"] = (
            LATENCY["q_hybrid_search_ms"] + LATENCY["q_rerank_ms"] + LATENCY["q_llm_generation_ms"]
        )
    print_latency_report()
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, parent_lookup = build_pipeline()
    evaluate_pipeline(search, reranker, parent_lookup)
    print(f"\nTotal: {time.time() - start:.1f}s")
