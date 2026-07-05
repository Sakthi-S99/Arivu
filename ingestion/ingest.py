"""
Arivu RAG — Ingestion pipeline.
Parse -> Chunk (fixed-size + overlap) -> Embed (BGE-M3) -> Store (Qdrant).

Usage:
    python ingest.py                 # ingest all files under DOCS_DIR
    python ingest.py --reset         # drop collection and re-ingest
"""

import os
import re
import sys
import glob
import json
import time
import uuid
import hashlib
import logging
import argparse
import subprocess

import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseVector,
    Filter, FieldCondition, MatchValue,
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),           # console
        logging.FileHandler(LOG_FILE),     # persistent log
    ],
)
log = logging.getLogger("arivu-ingest")

# Tracks which files are already ingested — enables resume after a crash
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ingest_state.json")

# Fixed namespace for deterministic chunk IDs — NEVER change this value
ARIVU_NS = uuid.UUID("00000000-0000-0000-0000-00000000a71b")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    OLLAMA_HOST, EMBED_MODEL, QDRANT_HOST, QDRANT_PORT, COLLECTION,
    VECTOR_SIZE, CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR,
    EMBED_BATCH_SIZE, EMBED_BATCH_SMALL_FILE, EMBED_BATCH_LARGE_FILE,
    SMALL_FILE_CHUNKS, LARGE_FILE_CHUNKS, EMBED_TIMEOUT,
    ENABLE_HYBRID, SPARSE_MODEL, MAX_CHUNK_CHARS, MAX_WORD_CHARS,
)


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text(path: str) -> tuple[str, dict]:
    """
    Extract raw text. Returns (text, meta).
    meta flags likely image-based or table-heavy pages for later OCR/table handling.
    """
    ext = path.lower().rsplit(".", 1)[-1]
    meta = {"pages": 0, "chars": 0, "chars_per_page": 0, "likely_image_based": False}

    if ext == "pdf":
        reader = PdfReader(path)
        pages = len(reader.pages)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        meta["pages"] = pages
        meta["chars"] = len(text)
        meta["chars_per_page"] = len(text) // pages if pages else 0
        # Text-based PDFs yield ~1500-3000 chars/page. Very low → images/scans.
        meta["likely_image_based"] = pages > 0 and meta["chars_per_page"] < 100
        return text, meta

    if ext in ("md", "txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        meta["chars"] = len(text)
        return text, meta

    return "", meta


# ── Chunking (fixed-size with overlap, word-based approximation) ───────────────
def _split_long_words(words: list[str]) -> list[str]:
    """
    Split any single 'word' longer than MAX_WORD_CHARS.
    Catches no-space/table extraction where a whole page becomes one token.
    """
    out = []
    for w in words:
        if len(w) <= MAX_WORD_CHARS:
            out.append(w)
        else:
            # Business Purpose: pathological extraction can glue a page into one token,
            # producing a chunk that blows past BGE-M3's token limit and stalls embedding.
            out.extend(w[i:i + MAX_WORD_CHARS] for i in range(0, len(w), MAX_WORD_CHARS))
    return out


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Fixed-size chunking with overlap.
    Word-based (~1.3 words/token) with two hard safety caps:
      - MAX_WORD_CHARS: split monster tokens from bad extraction
      - MAX_CHUNK_CHARS: cap chunk char length so token count stays under model limit
    """
    words = _split_long_words(text.split())
    if not words:
        return []

    chunks, start = [], 0
    step = size - overlap
    while start < len(words):
        window = words[start:start + size]
        chunk = " ".join(window)

        # Character ceiling — trim to whole words under MAX_CHUNK_CHARS
        if len(chunk) > MAX_CHUNK_CHARS:
            trimmed, length = [], 0
            for w in window:
                if length + len(w) + 1 > MAX_CHUNK_CHARS:
                    break
                trimmed.append(w)
                length += len(w) + 1
            chunk = " ".join(trimmed)
            # Advance only past what we actually consumed (keep overlap semantics)
            consumed = max(len(trimmed) - overlap, 1)
            if chunk.strip():
                chunks.append(chunk)
            start += consumed
            continue

        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks


# ── Embedding via Ollama (true batch) ─────────────────────────────────────────
def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts in a SINGLE request via Ollama's /api/embed.
    Returns one vector per input, order preserved.
    """
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def pick_batch_size(num_chunks: int) -> int:
    """
    Adaptive batch size: small files embed in big batches (fast),
    large files use small batches (avoid per-request timeout on Arc/16GB).
    """
    if num_chunks <= SMALL_FILE_CHUNKS:
        return EMBED_BATCH_SMALL_FILE
    if num_chunks >= LARGE_FILE_CHUNKS:
        return EMBED_BATCH_LARGE_FILE
    return EMBED_BATCH_SIZE


# ── Sparse embedding (hybrid) ─────────────────────────────────────────────────
_sparse_model = None


def _get_sparse_model():
    """Lazy-load BM25 sparse encoder — only when hybrid is enabled."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        log.info("Loading sparse model %s", SPARSE_MODEL)
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def sparse_batch(texts: list[str]):
    """Return list of (indices, values) sparse vectors for the batch."""
    return list(_get_sparse_model().embed(texts))


# ── GPU pre-flight check ──────────────────────────────────────────────────────
def check_gpu_backend():
    """
    Warn (non-fatal) if Ollama looks like it's serving the embed model on
    CPU-only. Embedding dominates ingestion time by orders of magnitude over
    extract/chunk/upsert (see ingest.log), so a silent CPU fallback turns a
    per-file job into hours instead of minutes — surface it up front instead
    of letting it get discovered halfway through a large run.
    """
    try:
        out = subprocess.run(
            ["journalctl", "-u", "ollama", "-n", "300", "--no-pager"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        matches = re.findall(r"inference compute.*?library=(\S+)", out)
        if matches:
            backend = matches[-1]  # most recent "starting runner" entry
            if backend == "cpu":
                log.warning(
                    "OLLAMA BACKEND = CPU — embedding will be the bottleneck "
                    "(minutes-to-hours per file, per ingest.log history). "
                    "If a GPU is available, enable it (e.g. OLLAMA_VULKAN=1 "
                    "for Intel Arc) and restart the ollama service first."
                )
            else:
                log.info("Ollama backend: %s (GPU-accelerated)", backend)
            return
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass

    log.info(
        "Could not determine Ollama compute backend (journalctl unavailable) — "
        "if embedding is unexpectedly slow, check GPU utilization manually."
    )


# ── Resume state ──────────────────────────────────────────────────────────────
def file_hash(path: str) -> str:
    """SHA-256 of file bytes — detects edits to already-ingested files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_state() -> dict:
    """Return {file_path: content_hash} of already-ingested files."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Old format (path-only, no hash). Migrate by stamping the
            # current hash so unchanged files aren't needlessly re-embedded;
            # edits made after this point are still detected normally.
            return {p: file_hash(p) for p in data if os.path.exists(p)}
        return data
    return {}


def save_state(done: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(done, f, indent=2, sort_keys=True)


def delete_source_points(client: QdrantClient, rel_source: str):
    """
    Remove all existing points for a source before re-ingesting it.
    Prevents stale orphaned chunks when content or chunking config changes
    shrink the chunk count for a file that still exists on disk.
    """
    if not client.collection_exists(COLLECTION):
        return
    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(must=[
            FieldCondition(key="source", match=MatchValue(value=rel_source))
        ]),
    )


# ── Orphan cleanup ────────────────────────────────────────────────────────────
def clean_orphans():
    """
    Remove points whose source file no longer exists on disk.
    Handles moved/renamed/deleted files without a full --reset.
    """
    docs_root = os.path.expanduser(DOCS_DIR)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(COLLECTION):
        log.warning("Collection does not exist — nothing to clean.")
        return

    # Collect distinct sources currently in Qdrant
    stored_sources = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=["source"],
            with_vectors=False,
        )
        for p in points:
            stored_sources.add(p.payload.get("source"))
        if offset is None:
            break

    # Find sources whose file is gone
    orphans = [s for s in stored_sources
               if not os.path.exists(os.path.join(docs_root, s))]

    if not orphans:
        log.info("No orphans found.")
        return

    # Delete points by source, and drop from resume state
    done = load_state()
    for src in orphans:
        client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=src))
            ]),
        )
        done.pop(os.path.join(docs_root, src), None)
        log.info("Removed orphan source: %s", src)

    save_state(done)
    log.info("Cleaned %d orphaned source(s).", len(orphans))


# ── Qdrant setup ──────────────────────────────────────────────────────────────
def get_client(reset: bool = False) -> QdrantClient:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    exists = client.collection_exists(COLLECTION)

    if reset and exists:
        client.delete_collection(COLLECTION)
        exists = False

    if not exists:
        if ENABLE_HYBRID:
            # Named dense vector + named sparse vector for hybrid search
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config={
                    "dense": VectorParams(
                        size=VECTOR_SIZE, distance=Distance.COSINE, on_disk=True,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            log.info("Created hybrid collection (dense + sparse).")
        else:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE, on_disk=True,
                ),
            )
            log.info("Created dense-only collection.")
    return client


# ── Main pipeline ─────────────────────────────────────────────────────────────
def ingest(reset: bool = False):
    docs_root = os.path.expanduser(DOCS_DIR)
    if not os.path.isdir(docs_root):
        log.error("Docs dir not found: %s", docs_root)
        return

    check_gpu_backend()

    client = get_client(reset=reset)

    # Reset also clears resume state
    done = {} if reset else load_state()
    if reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    patterns = ["**/*.pdf", "**/*.md", "**/*.txt"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(docs_root, p), recursive=True))

    if not files:
        log.warning("No documents found under %s", docs_root)
        return

    # Compare content hash, not just path — catches edits to already-ingested files
    pending = []
    for f in files:
        h = file_hash(f)
        if done.get(f) != h:
            pending.append((f, h))

    if not pending:
        log.info("All %d files already ingested and unchanged. Use --reset to rebuild.", len(files))
        return

    log.info("%d file(s) to ingest (%d unchanged).", len(pending), len(files) - len(pending))

    total_chunks = 0
    failed = []
    run_start = time.time()

    for n, (path, path_hash) in enumerate(pending, 1):
        fname = os.path.basename(path)
        log.info("[%d/%d] START %s", n, len(pending), fname)
        file_start = time.time()

        try:
            # Stage: extract
            t0 = time.time()
            text, meta = extract_text(path)
            t_extract = time.time() - t0

            if meta.get("likely_image_based"):
                log.warning("[%d/%d] LOW-TEXT %s — %d chars/page over %d pages; likely image/scanned, OCR needed later",
                            n, len(pending), fname, meta["chars_per_page"], meta["pages"])

            # Stage: chunk
            t0 = time.time()
            chunks = chunk_text(text)
            t_chunk = time.time() - t0

            if not chunks:
                log.warning("[%d/%d] SKIP  %s — no extractable text", n, len(pending), fname)
                done[path] = path_hash
                save_state(done)
                continue

            rel_source = os.path.relpath(path, docs_root)

            # Clear any previously ingested chunks for this source before
            # re-inserting — avoids orphaned tail chunks when content or
            # chunking config changes shrink the chunk count for a file
            # that still exists on disk.
            delete_source_points(client, rel_source)

            file_chunks = 0
            t_embed = 0.0
            t_upsert = 0.0
            batch_size = pick_batch_size(len(chunks))
            num_batches = (len(chunks) + batch_size - 1) // batch_size
            log.info("[%d/%d]   %d chunks → batch size %d (%d batches)",
                     n, len(pending), len(chunks), batch_size, num_batches)

            # Process in batches — embed + upsert per batch (memory-smooth, resumable)
            for b in range(0, len(chunks), batch_size):
                batch = chunks[b:b + batch_size]
                batch_no = b // batch_size + 1

                # Stage: embed (batch) — dense, plus sparse if hybrid
                te = time.time()
                vectors = embed_batch(batch)
                sparse_vecs = sparse_batch(batch) if ENABLE_HYBRID else [None] * len(batch)
                t_embed += time.time() - te

                if ENABLE_HYBRID:
                    points = [
                        PointStruct(
                            id=str(uuid.uuid5(ARIVU_NS, f"{rel_source}::{b + i}")),
                            vector={
                                "dense": vec,
                                "sparse": SparseVector(
                                    indices=sp.indices.tolist(),
                                    values=sp.values.tolist(),
                                ),
                            },
                            payload={"source": rel_source, "chunk_index": b + i, "text": chunk},
                        )
                        for i, (chunk, vec, sp) in enumerate(zip(batch, vectors, sparse_vecs))
                    ]
                else:
                    points = [
                        PointStruct(
                            id=str(uuid.uuid5(ARIVU_NS, f"{rel_source}::{b + i}")),
                            vector=vec,
                            payload={"source": rel_source, "chunk_index": b + i, "text": chunk},
                        )
                        for i, (chunk, vec) in enumerate(zip(batch, vectors))
                    ]

                # Stage: upsert (batch)
                tu = time.time()
                client.upsert(collection_name=COLLECTION, points=points)
                t_upsert += time.time() - tu

                file_chunks += len(points)
                log.info("[%d/%d]   batch %d/%d — %d chunks",
                         n, len(pending), batch_no, num_batches, len(points))

            total_chunks += file_chunks
            done[path] = path_hash
            save_state(done)

            log.info(
                "[%d/%d] OK    %s — %d chunks | extract %.2fs chunk %.2fs embed %.2fs upsert %.2fs | total %.1fs",
                n, len(pending), fname, file_chunks,
                t_extract, t_chunk, t_embed, t_upsert, time.time() - file_start,
            )

        except Exception as e:
            failed.append((fname, str(e)))
            log.error("[%d/%d] FAIL  %s — %s", n, len(pending), fname, e)
            continue

    run_elapsed = time.time() - run_start
    log.info("Done. %d new chunks in %.1fs. %d succeeded, %d failed.",
             total_chunks, run_elapsed, len(pending) - len(failed), len(failed))
    if failed:
        log.warning("Failed files:")
        for fname, err in failed:
            log.warning("  - %s: %s", fname, err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild collection")
    parser.add_argument("--clean-orphans", action="store_true",
                        help="Remove chunks whose source file no longer exists")
    args = parser.parse_args()

    if args.clean_orphans:
        clean_orphans()
    else:
        ingest(reset=args.reset)
