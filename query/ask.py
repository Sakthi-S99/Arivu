"""
Arivu RAG — Query pipeline.
Embed question -> Search Qdrant -> Build context -> LLM answer.

Usage:
    python ask.py "How do I configure the retry policy for background jobs?"
    python ask.py                    # interactive mode
    python ask.py --debug "term"     # dump raw candidate scores at each stage
"""

import os
import sys
import time
import logging

import requests
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    OLLAMA_HOST, EMBED_MODEL, LLM_MODEL, LLM_NUM_CTX,
    QDRANT_HOST, QDRANT_PORT, COLLECTION, TOP_K, SCORE_THRESHOLD,
    RETRIEVE_N, ENABLE_QUERY_EXPANSION, ENABLE_RERANK,
    RERANK_SCORE_THRESHOLD, ENABLE_HYBRID, SPARSE_MODEL,
)
from query.expander import expand_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arivu-ask")

_sparse_model = None


def _get_sparse_model():
    """Lazy-load sparse encoder for hybrid search."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        log.info("HYBRID → loading sparse model %s", SPARSE_MODEL)
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def embed(text: str) -> list[float]:
    """
    Embed a single query via Ollama /api/embed. Must match ingest model.
    keep_alive=0 evicts bge-m3 the instant this returns — on this iGPU's
    small shared Vulkan memory budget, a resident embed model steals GPU
    headroom from the generation model and forces it into a slower
    CPU/GPU split (see the generation-latency investigation).
    """
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": text, "keep_alive": 0},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _dump_candidates(label: str, items) -> None:
    """Debug dump: rank, score, source, and a text snippet for each candidate."""
    print(f"\n--- DEBUG: {label} ({len(items)} candidates) ---")
    for i, (point, score) in enumerate(items, 1):
        src = point.payload.get("source", "unknown")
        snippet = point.payload.get("text", "").replace("\n", " ")[:100]
        print(f"  {i:>2}. score={score:.4f}  [{src}]  {snippet}...")
    print("--- end DEBUG ---\n")


def retrieve(question: str, client: QdrantClient, debug: bool = False):
    """
    Pipeline: [expand] -> embed -> retrieve wide (hybrid|dense) -> [rerank] -> top-K.
    """
    # 1. Query expansion
    q = question
    if ENABLE_QUERY_EXPANSION:
        q = expand_query(question)
        if q != question:
            log.info("EXPAND → %s", q)

    # 2. Embed (dense)
    log.info("EMBED  → encoding query with %s", EMBED_MODEL)
    t0 = time.time()
    qvec = embed(q)
    log.info("EMBED  done — %d-dim in %.2fs", len(qvec), time.time() - t0)

    # 3. Retrieve wide
    t0 = time.time()
    if ENABLE_HYBRID:
        log.info("SEARCH → hybrid (dense+sparse) top-%d", RETRIEVE_N)
        sparse = next(_get_sparse_model().embed([q]))
        result = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                qmodels.Prefetch(query=qvec, using="dense", limit=RETRIEVE_N),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(indices=sparse.indices.tolist(),
                                               values=sparse.values.tolist()),
                    using="sparse", limit=RETRIEVE_N),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=RETRIEVE_N,
            with_payload=True,
        )
    else:
        log.info("SEARCH → dense top-%d", RETRIEVE_N)
        result = client.query_points(
            collection_name=COLLECTION,
            query=qvec, using=None,
            limit=RETRIEVE_N, with_payload=True,
        )
    hits = result.points
    log.info("SEARCH done — %d candidates in %.2fs", len(hits), time.time() - t0)
    if debug:
        _dump_candidates("search stage (pre-rerank/threshold)",
                          [(h, h.score) for h in hits])

    # 4. Rerank (cross-encoder) OR cosine threshold
    if ENABLE_RERANK and hits:
        from query.reranker import rerank
        t0 = time.time()
        all_ranked = rerank(q, hits)                          # full ranked list
        if debug:
            _dump_candidates("rerank stage (all candidates)", all_ranked)
        ranked = all_ranked[:TOP_K]
        if RERANK_SCORE_THRESHOLD is not None:
            ranked = [(p, s) for p, s in ranked if s >= RERANK_SCORE_THRESHOLD]
        log.info("RERANK done — kept %d in %.2fs (top score %.3f)",
                 len(ranked), time.time() - t0, ranked[0][1] if ranked else 0.0)
        return [(p, s, "rerank") for p, s in ranked]
    else:
        strong = [h for h in hits if h.score >= SCORE_THRESHOLD][:TOP_K]
        dropped = len(hits) - len(strong)
        if dropped:
            log.warning("FILTER — dropped %d below cosine %.2f", dropped, SCORE_THRESHOLD)
        return [(h, h.score, "cosine") for h in strong]


def build_context(hits) -> str:
    """
    Concatenate retrieved chunks. `hits` is a list of (point, score, label).
    Deduplicate identical text — keeps first occurrence, notes extra sources.
    """
    import hashlib

    seen = {}
    ordered = []

    for point, _score, _label in hits:
        txt = point.payload.get("text", "")
        src = point.payload.get("source", "unknown")
        digest = hashlib.sha256(txt.encode()).hexdigest()

        if digest in seen:
            seen[digest].append(src)
        else:
            seen[digest] = [src]
            ordered.append((digest, txt))

    blocks = []
    for digest, txt in ordered:
        sources = ", ".join(seen[digest])
        blocks.append(f"[Source: {sources}]\n{txt}")
    return "\n\n---\n\n".join(blocks)


_QUESTION_STARTERS = {
    "what", "how", "why", "when", "where", "who", "which", "whom", "whose",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "should", "will", "shall", "explain", "describe", "list", "summarize",
}


def _normalize_question(question: str) -> str:
    """
    Bare keywords/topic phrases ("Delinquency", "delinquency workflow
    configuration steps overview") aren't questions regardless of length —
    the strict prompt below has nothing to "answer" for them, so the model
    falls back to the not-found response even when the context is directly
    on topic. A word-count cutoff doesn't generalize (a 5-word topic phrase
    fails the same way a 1-word one does), so key off phrasing instead:
    rewrite anything that isn't already a question (no '?', doesn't open
    with a question/imperative word) into an explicit question it can act on.
    """
    stripped = question.strip()
    if not stripped or "?" in stripped:
        return question
    first_word = stripped.split()[0].lower()
    if first_word not in _QUESTION_STARTERS:
        return f'What does the documentation say about "{stripped}"?'
    return question


def ask_llm(question: str, context: str) -> str:
    question = _normalize_question(question)
    prompt = f"""You are answering strictly from the provided context.

Rules:
- Use ONLY facts present in the context below.
- Do NOT invent identifiers, class names, method names, or configuration values.
- If the context does not contain the answer, respond exactly: "The provided documents do not contain this information."
- Quote exact identifiers/values only if they appear verbatim in the context.
- Be concise: answer in as few sentences as fully answering the question requires.

Context:
{context}

Question: {question}

Answer:"""

    log.info("GENERATE → %s composing answer from %d chars context", LLM_MODEL, len(context))
    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": LLM_MODEL, "prompt": prompt, "stream": False,
            "options": {"num_ctx": LLM_NUM_CTX, "num_predict": 500},
        },
        timeout=1200,
    )
    resp.raise_for_status()
    out = resp.json()["response"].strip()
    log.info("GENERATE done — %d chars in %.2fs", len(out), time.time() - t0)
    return out


def answer(question: str, client: QdrantClient, debug: bool = False):
    hits = retrieve(question, client, debug=debug)
    if not hits:
        print(f"No chunks above score {SCORE_THRESHOLD}. Either the KB lacks this topic, or lower SCORE_THRESHOLD.")
        return

    context = build_context(hits)
    response = ask_llm(question, context)

    print("\n" + "=" * 70)
    print(response)
    print("=" * 70)
    print("\nSources:")
    seen = set()
    for point, score, label in hits:
        src = point.payload.get("source", "unknown")
        if src not in seen:
            print(f"  - {src}  ({label}: {score:.3f})")
            seen.add(src)


if __name__ == "__main__":
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    args = sys.argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]

    if args:
        answer(" ".join(args), client, debug=debug)
    else:
        print("Arivu RAG — interactive mode. Ctrl+C to exit.\n")
        try:
            while True:
                q = input("Q: ").strip()
                if q:
                    answer(q, client, debug=debug)
                    print()
        except KeyboardInterrupt:
            print("\nBye.")
