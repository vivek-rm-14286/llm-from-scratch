#!/usr/bin/env python3
"""Compile docs/*.md into HTML fragments + a manifest + a search index for the React app.

Uses Python-Markdown with the same pymdown-extensions the site was written against
(admonitions, details, tabs, math, mermaid fences, snippets, task lists, ...), so the
Markdown sources stay the single source of truth.

Outputs (all under app/src/generated/, git-ignored):
  manifest.json   tracks → modules → lessons, reference pages, per-page metadata
  pages/<id>.html one HTML fragment per page (H1 stripped; title lives in the manifest)
  search.json     section-level search records
"""
from __future__ import annotations

import html
import json
import posixpath
import re
import shutil
import sys
from pathlib import Path

import markdown
import yaml
from pymdownx import superfences

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
OUT = ROOT / "app" / "src" / "generated"


def md_instance() -> markdown.Markdown:
    return markdown.Markdown(
        extensions=[
            "abbr", "admonition", "attr_list", "def_list", "footnotes", "md_in_html", "tables",
            "toc", "pymdownx.arithmatex", "pymdownx.betterem", "pymdownx.caret", "pymdownx.details",
            "pymdownx.highlight", "pymdownx.inlinehilite", "pymdownx.keys", "pymdownx.mark",
            "pymdownx.snippets", "pymdownx.superfences", "pymdownx.tabbed", "pymdownx.tasklist",
        ],
        extension_configs={
            "toc": {"permalink": False},
            "pymdownx.arithmatex": {"generic": True},
            "pymdownx.highlight": {"pygments_lang_class": True},
            "pymdownx.snippets": {"base_path": [str(ROOT)], "check_paths": True},
            "pymdownx.superfences": {"custom_fences": [
                {"name": "mermaid", "class": "mermaid", "format": superfences.fence_code_format},
            ]},
            "pymdownx.tabbed": {"alternate_style": True},
            "pymdownx.tasklist": {"custom_checkbox": True},
        },
    )


def route_of(rel: str) -> str:
    """docs-relative path ('llm/attention.md') → app route ('llm/attention')."""
    p = rel[:-3] if rel.endswith(".md") else rel
    if p == "index":
        return ""
    if p.endswith("/index"):
        p = p[: -len("/index")]
    return p


def page_id(route: str) -> str:
    return route.replace("/", "__") or "home"


TAG_RE = re.compile(r"<[^>]+>")


def text_of(fragment: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", TAG_RE.sub(" ", fragment))).strip()


def main() -> int:
    cur = yaml.safe_load((DOCS / "curriculum.yml").read_text())
    routes: dict[str, str] = {}          # docs rel path → route
    meta: dict[str, dict] = {}           # route → page metadata

    tracks = []
    for t in cur["tracks"]:
        mods = []
        for mi, m in enumerate(t["modules"], 1):
            code = f'{t["code"]}{mi}'
            lessons = []
            for li, entry in enumerate(m["lessons"]):
                (label, rel), = entry.items()
                r = route_of(rel)
                routes[rel] = r
                lessons.append({"route": r, "label": label})
                meta[r] = {"src": rel, "label": label, "module": code, "moduleTitle": m["title"], "index": li}
            mods.append({"code": code, "title": m["title"], "blurb": m.get("blurb", ""), "lessons": lessons})
        tracks.append({"title": t["title"], "code": t["code"], "blurb": t["blurb"], "modules": mods})
    reference = []
    for entry in cur.get("reference", []):
        (label, rel), = entry.items()
        r = route_of(rel)
        routes[rel] = r
        reference.append({"route": r, "label": label})
        meta[r] = {"src": rel, "label": label, "module": None, "moduleTitle": None, "index": None}

    order = [l["route"] for t in tracks for m in t["modules"] for l in m["lessons"]] + [r["route"] for r in reference]

    # every markdown file must be in the curriculum (except the home page, which is a React view)
    for f in DOCS.rglob("*.md"):
        rel = f.relative_to(DOCS).as_posix()
        if rel not in routes and rel != "index.md":
            print(f"error: {rel} is not listed in docs/curriculum.yml", file=sys.stderr)
            return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "pages").mkdir(parents=True)

    href_re = re.compile(r'href="([^"]*)"')
    errors = []
    search = []
    for rel, r in routes.items():
        src = (DOCS / rel).read_text()
        md = md_instance()
        body = md.convert(src)

        m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
        title = text_of(m.group(1)) if m else meta[r]["label"]
        if m:
            body = body[: m.start()] + body[m.end():]

        base_dir = posixpath.dirname(rel)

        def fix(mo: re.Match) -> str:
            h = html.unescape(mo.group(1))
            if re.match(r"^[a-z]+:", h) or h.startswith("//"):
                return mo.group(0)
            if h.startswith("#"):
                return f'href="#/{r}{h}"'
            path, _, frag = h.partition("#")
            target = posixpath.normpath(posixpath.join(base_dir, path)) if path else rel
            if target.endswith("/") or not target.endswith(".md"):
                cand = target.rstrip("/") + "/index.md"
                cand2 = target.rstrip("/") + ".md"
                target = cand if cand in routes else cand2
            if target == "index.md":
                tr = ""
            elif target in routes:
                tr = routes[target]
            else:
                errors.append(f"{rel}: broken link {h}")
                return mo.group(0)
            return f'href="#/{tr}' + (f"#{frag}" if frag else "") + '"'

        body = href_re.sub(fix, body)
        # external links open in a new tab
        body = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener"', body)

        headings = [{"id": hid, "level": int(lv), "text": text_of(txt)}
                    for lv, hid, txt in re.findall(r'<h([23]) id="([^"]+)"[^>]*>(.*?)</h\1>', body, re.S)]
        (OUT / "pages" / f"{page_id(r)}.html").write_text(body)

        meta[r].update({"title": title, "id": page_id(r), "headings": headings,
                        "hasMath": "arithmatex" in body, "hasMermaid": 'class="mermaid"' in body,
                        "hasScript": "<script" in body})

        # search records: one per h2/h3 section
        parts = re.split(r'(<h[23] id="[^"]+"[^>]*>.*?</h[23]>)', body, flags=re.S)
        cur_id, cur_head, buf = "", title, []

        def flush():
            txt = text_of(" ".join(buf))
            if txt or cur_id == "":
                search.append({"r": r, "t": title, "h": cur_head if cur_id else "", "a": cur_id, "x": txt[:1200]})

        for p in parts:
            hm = re.match(r'<h[23] id="([^"]+)"[^>]*>(.*?)</h[23]>', p, re.S)
            if hm:
                flush()
                cur_id, cur_head, buf = hm.group(1), text_of(hm.group(2)), []
            else:
                p = re.sub(r"<pre.*?</pre>", " ", p, flags=re.S)
                p = re.sub(r'<(span|div) class="arithmatex">.*?</\1>', " ", p, flags=re.S)
                buf.append(p)
        flush()

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    manifest = {"site": cur["site"], "tracks": tracks, "reference": reference, "order": order, "pages": meta}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "search.json").write_text(json.dumps(search, ensure_ascii=False))
    print(f"built {len(routes)} pages, {len(search)} search records → {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
