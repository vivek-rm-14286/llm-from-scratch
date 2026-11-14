# LLM From Scratch

Source for https://vivek-rm-14286.github.io/llm-from-scratch/ — a hands-on curriculum for building, pretraining, post-training, fine-tuning and serving language models, RAG systems and decision models.

- `docs/` — site content (MkDocs Material)
- `code/` — runnable code for the four projects

## Local preview

```bash
uv venv && uv pip install -r requirements.txt
.venv/bin/mkdocs serve
```

Deployed automatically to GitHub Pages by `.github/workflows/deploy.yml` on push to `main` (set *Settings → Pages → Source* to **GitHub Actions**).
