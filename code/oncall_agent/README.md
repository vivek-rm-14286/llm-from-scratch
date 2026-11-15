# Project 8: On-Call Agent (tools, MCP, guardrails)

An agent for the on-call engineer of the fictional product **Nimbus Monitor**. It takes an incident
(alerts, a ticket or a question), routes and triages it, reads runbooks, checks simulated metrics
and logs, proposes actions, and asks a human before anything irreversible. The project is the
agent loop, the approval gate, the guardrails, an MCP server and a trajectory-level evaluation
suite. Everything runs offline with two deterministic fake LLMs; a real OpenAI-compatible
endpoint is optional.

Full walkthrough: https://vivek-rm-14286.github.io/llm-from-scratch/#/projects/oncall-agent

## Layout

```text
oncall/      shared code, already written: simulator, tools, adapters, fake LLMs, scenarios
starter/     signatures, docstrings and TODO(1)..TODO(12): write the bodies yourself
solution/    the reference implementation (same file names)
tests/       53 pytest tests, run against either package
ci/          agent-eval-gate.yml, a sample GitHub Actions workflow
```

| File | What it does |
|---|---|
| `starter/guardrails.py` | scan, quarantine and wrap tool output; least-privilege tool sets; grounding check; report validation |
| `starter/approval.py` | `ApprovalGate`: irreversible tools run only with an approval, and it fails closed |
| `starter/agent.py` | `run_agent`: the loop, step limit, token and cost budget, malformed-call feedback, JSONL trace |
| `starter/evaluate.py` | trajectory checks per scenario, aggregation, the release gate |
| `starter/mcp_server.py` | an MCP server (`mcp` 2.x, `MCPServer`) with the five read-only tools |
| `oncall/simulator.py` | seeded fleet of 8 services: metrics, logs, faults, and the log of executed actions |
| `oncall/tools.py`, `oncall/schemas.py` | the eight tools and their strict Pydantic schemas |
| `oncall/adapters.py` | adapters to Projects 3, 5 and 6, each with a labelled local fallback |
| `oncall/runbooks.py` | local fallback runbook corpus with access levels |
| `oncall/fake_llm.py` | `CarefulPolicy`, `NaivePolicy`, `LoopingPolicy`, `ScriptedLLM` |
| `oncall/llm.py` | `OpenAIChat`: any OpenAI-compatible `/chat/completions` endpoint, standard library only |
| `oncall/scenarios.py` | the 30 scripted incidents and their expected outcomes |
| `run_agent.py` | work one incident and print the trace |
| `run_eval.py` | run the suite, print the metrics, apply the gate |
| `run_mcp_client.py` | start the MCP server over stdio and call it from a client |

## Setup

```bash
cd code/oncall_agent
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -r requirements.txt
```

No GPU, no download, no other project needed.

```bash
pytest tests --impl solution     # 53 passed
pytest tests --impl starter      # 46 failed, 7 passed: the failures are the to-do list
```

For `pytest`, `--impl` defaults to `starter`, so plain `pytest tests` is the to-do list (the run scripts default to `solution`). The 7 tests that pass on the untouched starter cover the shared
code in `oncall/`.

## Run it

```bash
python run_agent.py --scenario N01                      # you approve or deny at the prompt
python run_agent.py --scenario I06 --llm naive --guardrails off --approve all
python run_agent.py --kind question --access public --text "How do I mute a monitor?"

python run_eval.py -v                                   # careful fake LLM, guardrails on
python run_eval.py --matrix                             # both fake LLMs x guardrails x approver
python run_eval.py --max-violations 0 --min-success 0.9 --max-forbidden-rate 0    # the CI gate

python run_mcp_client.py                                # MCP server + client over stdio
```

Every command takes `--impl starter` once your tests pass.

## A real model

Any OpenAI-compatible endpoint that supports tool calls:

```bash
ollama pull qwen2.5:7b
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b     # LLM_API_KEY for hosted endpoints
python run_agent.py --scenario N04 --llm endpoint
python run_eval.py --policy endpoint --trace-dir traces -v
```

Set `price_in_per_m` and `price_out_per_m` in `AgentConfig` to your endpoint's prices for a real
cost per task.

## The other projects

Each adapter reports which backend answered in a `backend` field. `--backends fallback` (the
default for the evaluation suite) never touches the other projects, `auto` uses them when
available and says why when not, `real` raises when one is missing.

| Tool | Real backend | How to enable | Local fallback |
|---|---|---|---|
| `route_ticket` | Project 3, `code/ticket_router`, `solution.router.load_router` in a subprocess | `TICKET_ROUTER_PYTHON=/path/to/that/venv/bin/python` (it needs numpy and, for trained models, torch) | keyword counts |
| `triage_incident` | Project 6, `code/incident_triage`, `inference.triage` against a served model | `TRIAGE_BASE_URL=http://localhost:8000/v1 TRIAGE_MODEL=triage` | rule-based triage that fills the same schema |
| `search_runbooks` | Project 5, `code/support_rag`: `POST /ask` or the in-process `Retriever` | `RAG_URL=http://localhost:8000/ask`, or `RUNBOOK_BACKEND=project5` with Project 5's requirements installed and its index built | `oncall/runbooks.py`, keyword search with access levels |

```bash
RUNBOOK_BACKEND=project5 TICKET_ROUTER_PYTHON=../ticket_router/.venv/bin/python \
  python run_agent.py --kind ticket --text "We were charged twice this month" --backends auto --approve none
```

## CI

`ci/agent-eval-gate.yml` is a sample workflow. Copy it to `.github/workflows/` to enable it. It
runs the tests, requires zero approval violations in all eight configurations, and gates on task
success, forbidden actions and attack success rate.
