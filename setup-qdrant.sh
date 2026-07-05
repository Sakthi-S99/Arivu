#!/bin/bash
# Arivu RAG — Qdrant vector DB setup
# On-demand only (restart=no)

docker run -d \
  --name qdrant \
  --restart=no \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

echo "Qdrant started."
echo "REST API:  http://localhost:6333"
echo "Dashboard: http://localhost:6333/dashboard"
