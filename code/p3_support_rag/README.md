# Project 3: Support Engineer Copilot (production-style RAG)

RAG over product docs (markdown) and resolved support tickets for the fictional product
**Nimbus Monitor**, with hybrid search (Qdrant dense vectors + BM25, fused with RRF), a cross-encoder
reranker, per-document access levels, cited answers from any OpenAI-compatible LLM, a FastAPI
service, and an evaluation harness.

Full walkthrough: https://vivek-rm-14286.github.io/llm-from-scratch/projects/p3-rag/

## Files

| File | Purpose |
|---|---|
| `settings.py` | All configuration (env-overridable), demo API keys -> access levels |
| `ingest.py` | Parse front matter, heading-aware chunking, embed, index in Qdrant + write `chunks.jsonl` |
| `retrieve.py` | Dense (Qdrant, ACL filter) + BM25 (ACL filter) -> RRF -> cross-encoder rerank |
| `generate.py` | Prompt with numbered sources, citations, abstention, citation validation |
| `app.py` | FastAPI: `POST /ask`, `POST /ingest`, `GET /health` |
| `eval.py` | recall@k / MRR per retrieval mode, ACL leak check, LLM-judge faithfulness |
| `sample_data/` | 9 markdown docs (public / internal / restricted), 15 tickets, 20 golden questions |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# LLM: any OpenAI-compatible endpoint. Default is local Ollama:
ollama pull qwen2.5:7b-instruct
# or: export LLM_BASE_URL=https://api.openai.com/v1 LLM_MODEL=<model> LLM_API_KEY=<key>

python ingest.py                                   # builds ./index (Qdrant local mode + chunks.jsonl)
python retrieve.py "What does NM-4021 mean?" --access public
python eval.py                                     # retrieval metrics for 4 modes
python eval.py --generate                          # + generation metrics (needs the LLM)

uvicorn app:app --port 8080                        # stop eval.py first: local mode allows one process
curl -s localhost:8080/ask -H 'X-API-Key: support-demo-key' -H 'Content-Type: application/json' \
     -d '{"question": "Can support temporarily raise a customer API rate limit?"}' | python -m json.tool
```

Demo keys (override with `NIMBUS_API_KEYS` as JSON): `public-demo-key` (public),
`support-demo-key` (internal), `security-demo-key` (restricted; also allowed to call `/ingest`).

To use a Qdrant server instead of local mode: `docker run -p 6333:6333 qdrant/qdrant` and
`export NIMBUS_QDRANT_URL=http://localhost:6333`.
