"""
Arivu RAG — Reranking.
Cross-encoder re-scores retrieved candidates against the query.
Far more accurate than cosine for final ranking. No re-ingest needed.

Model loads lazily on first use (fastembed downloads ONNX weights once, then caches).
"""

import logging
from config.settings import RERANK_MODEL

log = logging.getLogger("arivu-rerank")

_encoder = None


def _get_encoder():
    """Lazy-load the cross-encoder — avoids startup cost when rerank disabled."""
    global _encoder
    if _encoder is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        log.info("RERANK → loading model %s (first run downloads weights)", RERANK_MODEL)
        _encoder = TextCrossEncoder(model_name=RERANK_MODEL)
    return _encoder


def rerank(query: str, hits, top_k: int):
    """
    Re-score `hits` (Qdrant points) against `query`, return top_k.
    Returns list of (point, rerank_score) tuples — ScoredPoint is immutable,
    so scores are carried alongside rather than set on the object.
    """
    if not hits:
        return []

    encoder = _get_encoder()
    docs = [h.payload.get("text", "") for h in hits]
    scores = list(encoder.rerank(query, docs))   # one score per doc

    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
