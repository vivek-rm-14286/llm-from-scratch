# LLM From Scratch

Source for https://vivek-rm-14286.github.io/llm-from-scratch/ — a hands-on textbook for building, pretraining, post-training, fine-tuning and serving language models, RAG systems, agents and decision models, with four capstone projects.

- `docs/` — lesson content in Markdown (admonitions, tabs, math, Mermaid, snippets), plus `docs/curriculum.yml` (tracks → modules → lessons)
- `app/` — the site: Vite + React 19 + TypeScript + Tailwind 4, HashRouter. `app/scripts/build_content.py` compiles the Markdown into per-page HTML, a manifest and a search index
- `code/` — runnable code for the four projects

## Local development

```bash
uv venv && uv pip install -r requirements.txt     # Markdown compiler deps
cd app && npm install
PATH="../.venv/bin:$PATH" npm run dev             # compiles docs/ then starts Vite
```

Add a page: write `docs/<section>/<page>.md`, list it in `docs/curriculum.yml`. The build fails on broken internal links or unlisted pages.

Deployed by `.github/workflows/deploy.yml` on push to `main` (Pages source: GitHub Actions).
