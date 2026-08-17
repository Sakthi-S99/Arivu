# Arivu

Local, privacy-first RAG pipeline. Ollama + Qdrant + BGE-M3. No cloud, no data leaves the machine.

> Source documents (PDFs, notes) are **never** committed. Only code lives here.

---

## Stack

| Component | Role |
|---|---|
| Ollama | LLM + embedding runtime |
| BGE-M3 | Dense embeddings (1024-dim) |
| Qdrant/bm25 (fastembed) | Sparse lexical embeddings for hybrid search |
| Qdrant | Vector store (on-disk, hybrid dense + sparse) |
| ms-marco-MiniLM-L-6-v2 (fastembed) | Cross-encoder reranker |
| Llama 3.1 8B | Answer generation |

---

## Retrieval pipeline

```
question --> [acronym expansion] --> embed (BGE-M3)
          --> hybrid search (dense + BM25, RRF fusion, top 40)
          --> [cross-encoder rerank --> top 6]
          --> grounded LLM answer + cited sources
```

Query expansion and reranking are config toggles (`ENABLE_QUERY_EXPANSION`, `ENABLE_RERANK`) — flip them off to fall back to plain dense cosine search. Hybrid search (`ENABLE_HYBRID`) requires a `--reset` since it changes the collection schema.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

bash setup-qdrant.sh                # first-time Qdrant container
```

Add to `~/.bashrc`:
```bash
export ARIVU_HOME="$HOME/projects/arivu"
alias qdrant-start='docker start qdrant'
alias qdrant-stop='docker stop qdrant'
alias arivu-ingest='python $ARIVU_HOME/ingestion/ingest.py'
alias arivu-ask='python $ARIVU_HOME/query/ask.py'
alias arivu-clean='python $ARIVU_HOME/ingestion/ingest.py --clean-orphans'
```

Ollama is expected to run as a long-lived service (e.g. `systemctl enable --now ollama`), not started per-session.

---

## Usage

```bash
# Start services
qdrant-start

# Put documents in ~/ai-knowledge-base/, then:
arivu-ingest                 # ingest new/changed files (resume-safe, content-hash aware)
arivu-ingest --reset         # full rebuild
arivu-clean                  # remove chunks for deleted/moved files

# Query
arivu-ask "How do I configure the retry policy for background jobs?"
arivu-ask                    # interactive mode

# Evaluate groundedness (hallucination gate)
python eval/evaluate.py
python eval/evaluate.py --verbose
```

---

## Structure

```
arivu/
├── config/
│   ├── settings.py            # central config
│   └── acronyms.local.json    # optional, gitignored — your own domain acronyms
├── ingestion/
│   └── ingest.py              # parse -> chunk -> embed (dense+sparse) -> store
├── query/
│   ├── ask.py                 # expand -> embed -> hybrid search -> rerank -> answer
│   ├── expander.py            # acronym/synonym query expansion
│   └── reranker.py            # cross-encoder reranking
├── eval/
│   ├── evaluate.py            # groundedness / hallucination eval
│   └── questions.json         # eval question set — replace with your own
├── setup-qdrant.sh            # Qdrant container
├── LICENSE
└── requirements.txt
```

---

## Config

Edit `config/settings.py`:

| Setting | Default | Notes |
|---|---|---|
| `EMBED_MODEL` | bge-m3:latest | 1024-dim; must match at ingest + query |
| `LLM_MODEL` | llama3.1:8b | Answer generation |
| `LLM_NUM_CTX` | 6144 | Context window — must cover the prompt + `TOP_K` chunks or Ollama silently truncates. A guard in `ask_llm()` logs a warning if a prompt gets close to this limit |
| `EVAL_JUDGE_MODEL` | = `LLM_MODEL` | Override to a different model for independent faithfulness grading |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 450 / 70 | Words per chunk / overlap between chunks — smaller chunks reduce the odds one chunk spans multiple unrelated subsections |
| `RETRIEVE_N` | 40 | Candidates fetched before reranking |
| `TOP_K` | 6 | Final chunks kept after rerank — tuned against `LLM_NUM_CTX`; raising it risks silent truncation, watch the log warning |
| `ENABLE_HYBRID` | True | Dense + BM25 sparse search with RRF fusion — needs `--reset` to change |
| `ENABLE_RERANK` | True | Cross-encoder rerank of the retrieved pool |
| `ENABLE_QUERY_EXPANSION` | True | Acronym expansion — no re-ingest needed |
| `RERANK_SCORE_THRESHOLD` | -6.0 | Cross-encoder logit floor, applied after slicing to `TOP_K` (prunes weak survivors, doesn't widen recall); MiniLM logits run negative even for good matches, so this only cuts candidates far below any observed real match |
| `SCORE_THRESHOLD` | 0.4 | Cosine floor, only used when `ENABLE_RERANK` is off |

Advanced knobs (adaptive embed batch sizing, hard chunk-size ceilings, timeouts) live further down in the same file with inline comments.

Domain-specific query-expansion acronyms go in `config/acronyms.local.json` (gitignored) rather than in code, so your vocabulary never has to leave your machine even though the pipeline code is public.

---

## Key Behaviors

- **Resume-safe** — checkpoints per file (`.ingest_state.json`); crash resumes cleanly
- **Idempotent** — deterministic chunk IDs; re-ingest overwrites, no duplicates
- **Change-aware** — tracks a content hash per file, so editing a file in place (without renaming) triggers a clean re-embed; old chunks for that source are cleared before the new ones are inserted
- **Noise-stripped** — table-of-contents dot-leader lines and repeating page header/footer boilerplate are filtered out before chunking, so they don't pollute chunk embeddings across the corpus
- **GPU pre-flight check** — `arivu-ingest` checks whether Ollama reports a GPU backend and warns loudly if it's silently running CPU-only, since embedding dominates ingestion time by orders of magnitude over parsing/chunking/upserting. On an Intel iGPU, both `OLLAMA_VULKAN=1` *and* (on newer Ollama builds) `OLLAMA_IGPU_ENABLE=1` need to be set in the `ollama` systemd unit, or Ollama silently drops to CPU-only
- **Fails loudly, not silently** — a Qdrant connection failure at startup is caught and logged with an actionable message instead of crashing with an unlogged traceback; a Qdrant outage previously looked like `ingest.log` had just stopped mid-run
- **On-disk vectors** — low RAM footprint on 16GB
- **Logged** — console + `ingest.log`; failed files skip, don't halt the run

> Note: changing `CHUNK_SIZE`/`CHUNK_OVERLAP` does **not** retroactively re-chunk files whose content hasn't changed (hash is unchanged, so they're skipped). Run `arivu-ingest --reset` after tuning chunking to apply it to the whole corpus.

---

## Requirements

- Ollama with your embed model (`bge-m3`) and LLM (`llama3.1:8b`, or whatever you set `LLM_MODEL` to) pulled
- Docker (Qdrant)
- Python 3.10+
- First run downloads and caches the fastembed ONNX weights (BM25 sparse model + cross-encoder reranker) — no Ollama pull needed for those

---

## License

MIT — see [LICENSE](LICENSE).
