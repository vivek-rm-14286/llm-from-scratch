# Project 3: Support Engineer Copilot (production-style RAG)

RAG over product docs (markdown) and resolved support tickets for the fictional product
**Nimbus Monitor**, with hybrid search (Qdrant dense vectors + BM25, fused with RRF), a cross-encoder
reranker, per-document access levels, cited answers from any OpenAI-compatible LLM, guardrails,
a FastAPI service, an evaluation harness, a red-team script and a sample CI gate.

Full walkthrough: https://vivek-rm-14286.github.io/llm-from-scratch/#/projects/p3-rag

## Files

| File | Purpose |
|---|---|
| `settings.py` | All configuration (env-overridable), demo API keys -> access levels |
| `ingest.py` | Parse front matter, heading-aware chunking, embed, index in Qdrant + write `chunks.jsonl` |
| `retrieve.py` | Dense (Qdrant, ACL filter) + BM25 (ACL filter) -> RRF -> cross-encoder rerank |
| `generate.py` | Prompt with numbered sources, citations, abstention, citation validation |
| `guardrails.py` | Input check, scan of retrieved chunks, output link/image/prompt-leak filter |
| `app.py` | FastAPI: `POST /ask`, `POST /ingest`, `GET /health` |
| `eval.py` | recall@k / MRR per retrieval mode and question type, ACL leak check, CI gate, LLM-judge faithfulness |
| `finetune_embedder.py` | Contrastive fine-tuning of the embedder with hard negatives; recall before/after on held-out questions |
| `redteam.py` | Fixed attack set against a poisoned index; attack success rate without and with guardrails |
| `build_sample_data.py` | Regenerates the reference pages, `tickets.jsonl` and `golden.jsonl`; validates the golden labels |
| `sample_data/` | 35 markdown docs (public / internal / restricted), 64 tickets, 316 chunks, 89 golden questions tagged by type |
| `redteam/` | Planted content (`poisoned/`) and the attack set (`attacks.jsonl`); never part of the normal index |
| `ci/` | `eval-gate.yml` (sample GitHub Actions workflow) and `baseline.json` (metrics the gate compares against) |

## Run it

No GPU needed. The embedding model (about 130 MB) and reranker (about 90 MB) download on first use.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python ingest.py                                   # builds ./index (Qdrant local mode + chunks.jsonl)
python retrieve.py "What does NM-4012 mean?" --access public --mode dense
python retrieve.py "What does NM-4012 mean?" --access public --mode sparse
python eval.py                                     # retrieval metrics for 4 modes, by question type
python redteam.py --fake-llm                       # attack success rate, no LLM needed

# Everything below needs an LLM: any OpenAI-compatible endpoint. Default is local Ollama:
ollama pull qwen2.5:7b
# or: export LLM_BASE_URL=https://api.openai.com/v1 LLM_MODEL=<model> LLM_API_KEY=<key>

python eval.py --generate                          # + generation metrics
python redteam.py                                  # attack success rate against the real model
uvicorn app:app --port 8080                        # stop eval.py first: local mode allows one process
curl -s localhost:8080/ask -H 'X-API-Key: support-demo-key' -H 'Content-Type: application/json' \
     -d '{"question": "Can support temporarily raise a customer API rate limit?"}' | python -m json.tool
```

Demo keys (override with `NIMBUS_API_KEYS` as JSON): `public-demo-key` (public),
`support-demo-key` (internal), `security-demo-key` (restricted; also allowed to call `/ingest`).

To use a Qdrant server instead of local mode: `docker run -p 6333:6333 qdrant/qdrant` and
`export NIMBUS_QDRANT_URL=http://localhost:6333`.

`NIMBUS_GUARDRAILS=0` turns the guardrails off.
