# Content style guide (for all authors of this site)

Site: custom React app in `app/` that compiles `docs/*.md` (Python-Markdown + pymdown-extensions). Every page must be listed in `docs/curriculum.yml`. Runnable project code in `code/<project>/`.
Live URL: https://vivek-rm-14286.github.io/llm-from-scratch/  Repo: https://github.com/vivek-rm-14286/llm-from-scratch
Link to repo code as https://github.com/vivek-rm-14286/llm-from-scratch/tree/main/code/<dir>

## Audience & voice
- A software engineer, comfortable with Python, rusty on math, wants to build models from scratch.
- Teach from first principles: intuition -> minimal math -> working PyTorch code -> practical gotchas.
- Direct, concrete, no fluff, no hype. Second person ("you"). Minimal emoji.
- Current date is November 2026. Reference well-established models/papers (GPT-2/3, Llama 3.x, Qwen2.5/Qwen3, Gemma 3, SmolLM2/3, Phi, DeepSeek-V3/R1, OLMo 2, Mistral). Never invent benchmark numbers, paper titles, or APIs. If unsure of an exact figure, say "roughly" or omit it.
- Library code: use stable, widely-documented APIs (torch, transformers, datasets, peft, trl, bitsandbytes, sentence-transformers, qdrant-client, gymnasium). Pin or state versions where APIs have shifted (e.g. TRL SFTConfig/SFTTrainer).

## Page structure (concept pages)
1. `# Title` then a one-paragraph hook.
2. `!!! abstract "What you'll learn"` bullet list.
3. Sections with `##`/`###`. Use intuition first, then math, then code.
4. `!!! warning "Common pitfalls"` near the end.
5. `## Build it` (core concept pages): 1-3 hands-on tasks. Each states the starting point, what to build, and the expected result (a number, a printed output, or a check that must pass) so the learner knows when they are done. Put a short solution sketch in a collapsible `??? success "Solution sketch"`. Only include tasks you have actually solved.
6. `## Check your understanding` with 3-5 collapsible Q&As: `??? question "Q..."` with the answer indented inside.
7. `## Further reading` - real papers (arXiv links you are confident about) and docs.
8. No manual "Next:" link — the app renders previous/next cards.
Concept pages: ~1500-3500 words. Project pages: long, with complete, runnable code.

## Markdown features available
- Admonitions: `!!! note|tip|warning|danger|example|info|abstract "Title"`; collapsible `??? note "Title"`.
- Tabs: `=== "PyTorch"` (content indented 4 spaces).
- Math: inline `$...$`, display `$$...$$` on their own lines (KaTeX). Avoid `$` for currency - write "USD 50" instead.
- Diagrams: ```mermaid fenced blocks (flowchart LR/TD, sequenceDiagram). Keep labels simple; quote labels with special chars: A["Q·Kᵀ / √d"].
- Code: ```python title="model.py" ; line highlights `hl_lines="3 4"`; annotations `# (1)!` followed by a numbered list.
- Cards grid: `<div class="grid cards" markdown>` ... `</div>`.
- Task lists `- [ ] item`. Keys `++ctrl+c++`. Footnotes `[^1]`.

## Links
- Internal links are RELATIVE .md paths, e.g. from docs/llm/attention.md -> `[Transformer](transformer.md)`, `[RAG](../rag/fundamentals.md)`. They are validated at build time. Only link to pages listed in docs/curriculum.yml.
- Only write the files assigned to you.
- No provenance wording ("according to X docs", "as captured from..."). Write plain explanatory content; papers go in "Further reading".
- Write impersonally: no "we measured" / "when we tested"; state the result ("On a laptop CPU this takes about 80 ms").
- US spelling. Units: "TFLOP/s", "GB" (say GiB only when the distinction matters), currency as "USD 50".
- The app adds a reading time, an "On this page" rail, previous/next cards and a "Used in" box (from `used_in:` in docs/curriculum.yml), so pages do not need to repeat those.
