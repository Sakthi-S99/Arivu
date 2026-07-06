"""Arivu RAG — central config. Single source of truth."""

# Ollama
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "bge-m3:latest"          # 1024-dim embeddings
LLM_MODEL   = "llama3.1:8b"   # answer generation
LLM_NUM_CTX = 6144                     # context window — must fit prompt + TOP_K chunks or Ollama silently truncates
                                        # (real contexts run ~5-6K tokens; trimmed from 8192 to free iGPU KV-cache memory)

# Eval — override to a different model so faithfulness grading isn't the
# generator grading its own homework. Defaults to LLM_MODEL if unset.
EVAL_JUDGE_MODEL = LLM_MODEL

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION  = "arivu_kb"
VECTOR_SIZE = 1024                      # BGE-M3 output dimension
DISTANCE    = "Cosine"

# Chunking (fixed-size with overlap)
CHUNK_SIZE    = 700                     # target tokens per chunk
CHUNK_OVERLAP = 100                     # token overlap between chunks
MAX_CHUNK_CHARS = 4000                  # hard char ceiling — keeps chunk under BGE-M3 token limit
MAX_WORD_CHARS  = 1000                  # split monster tokens from no-space/table extraction

# Ingestion batching (adaptive — scales with file size)
EMBED_BATCH_SIZE = 32                   # default / fallback batch
EMBED_BATCH_SMALL_FILE = 128            # batch when file has few chunks
EMBED_BATCH_LARGE_FILE = 16             # batch when file has many chunks
SMALL_FILE_CHUNKS = 200                 # <= this many chunks → small-file batch
LARGE_FILE_CHUNKS = 800                 # >= this many chunks → large-file batch
EMBED_TIMEOUT = 600                     # seconds per embed request

# Retrieval (two-stage: retrieve wide, rerank narrow)
RETRIEVE_N = 20                        # candidates fetched before reranking
TOP_K = 5                              # final chunks kept after rerank
SCORE_THRESHOLD = 0.4                  # cosine floor (dense-only mode)

# Query expansion (acronym/synonym) — no re-ingest needed
ENABLE_QUERY_EXPANSION = True

# Reranking (cross-encoder) — no re-ingest needed
ENABLE_RERANK = True
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # light ONNX cross-encoder
# Cross-encoder logit floor. MiniLM logits are frequently negative even for
# plausible matches (observed real hits as low as ~1.4), so 0.0 would silently
# drop them — -6.0 only filters candidates far below any observed real match
# (e.g. an off-topic query scored -7.6), cutting context noise without risking
# true positives.
RERANK_SCORE_THRESHOLD = -6.0

# Hybrid search (dense + sparse BM25) — REQUIRES re-ingest (--reset)
ENABLE_HYBRID = True
SPARSE_MODEL = "Qdrant/bm25"           # lexical sparse vectors, catches exact terms

# Local document source — NEVER committed to git
DOCS_DIR = "~/ai-knowledge-base"       # PDFs, markdown, notes live here
