# Arivu

Local, privacy-first RAG pipeline. Ollama + Qdrant + BGE-M3. No cloud, no data leaves the machine.

> Source documents (PDFs, notes) are **never** committed. Only code lives here.

---

## Stack

| Component | Role |
|---|---|
| Ollama | LLM + embedding runtime |
| BGE-M3 | Embeddings (1024-dim) |
| Qwen3-Coder | Answer generation |
| Qdrant | Vector store (on-disk) |

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

---

## Usage

```bash
# Start services
ollama-local && qdrant-start

# Put documents in ~/ai-knowledge-base/, then:
arivu-ingest                 # ingest new files (resume-safe)
arivu-ingest --reset         # full rebuild
arivu-clean                  # remove chunks for deleted/moved files

# Query
arivu-ask "How do I configure the retry policy for background jobs?"
arivu-ask                    # interactive mode
```

---

## Structure

```
arivu/
├── config/settings.py       # central config
├── ingestion/ingest.py      # parse → chunk → embed → store
├── query/ask.py             # embed → search → context → answer
├── setup-qdrant.sh          # Qdrant container
└── requirements.txt
```

---

## Config

Edit `config/settings.py`:

| Setting | Default | Notes |
|---|---|---|
| `EMBED_MODEL` | bge-m3:latest | 1024-dim; must match at ingest + query |
| `LLM_MODEL` | qwen3-coder:latest | Answer generation |
| `CHUNK_SIZE` | 700 | Words per chunk |
| `CHUNK_OVERLAP` | 100 | Context bridge between chunks |
| `TOP_K` | 5 | Chunks retrieved per query |

---

## Key Behaviors

- **Resume-safe** — checkpoints per file (`.ingest_state.json`); crash resumes cleanly
- **Idempotent** — deterministic chunk IDs; re-ingest overwrites, no duplicates
- **Change-aware** — tracks a content hash per file, so editing a file in place (without renaming) triggers a clean re-embed; old chunks for that source are cleared before the new ones are inserted
- **On-disk vectors** — low RAM footprint on 16GB
- **Logged** — console + `ingest.log`; failed files skip, don't halt the run

> Note: changing `CHUNK_SIZE`/`CHUNK_OVERLAP` does **not** retroactively re-chunk files whose content hasn't changed (hash is unchanged, so they're skipped). Run `arivu-ingest --reset` after tuning chunking to apply it to the whole corpus.

---

## Requirements

- Ollama with `bge-m3` and `qwen3-coder` pulled
- Docker (Qdrant)
- Python 3.10+
