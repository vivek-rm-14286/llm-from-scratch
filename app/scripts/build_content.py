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

    final_out = OUT
    tmp_out = OUT.with_name(OUT.name + ".tmp")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    (tmp_out / "pages").mkdir(parents=True)

    href_re = re.compile(r'href="([^"]*)"')
    errors = []
    search = []

    # pass 1: compile every page, pull out the title, collect heading/anchor ids
    bodies: dict[str, str] = {}
    ids: dict[str, set[str]] = {}
    for rel, r in routes.items():
        body = md_instance().convert((DOCS / rel).read_text())
        m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
        meta[r]["title"] = text_of(m.group(1)) if m else meta[r]["label"]
        if m:
            body = body[: m.start()] + body[m.end():]
        bodies[rel] = body
        ids[r] = set(re.findall(r'\bid="([^"]+)"', body))

    # pass 2: rewrite and validate links (page and anchor), build metadata and search records
    for rel, r in routes.items():
        body = bodies[rel]
        title = meta[r]["title"]
        base_dir = posixpath.dirname(rel)

        def fix(mo: re.Match) -> str:
            h = html.unescape(mo.group(1))
            if re.match(r"^[a-z]+:", h) or h.startswith("//"):
                return mo.group(0)
            path, _, frag = h.partition("#")
            if not path:
                tr = r
            else:
                target = posixpath.normpath(posixpath.join(base_dir, path))
                if target.endswith("/") or not target.endswith(".md"):
                    cand = target.rstrip("/") + "/index.md"
                    target = cand if cand in routes else target.rstrip("/") + ".md"
                if target == "index.md":
                    return 'href="#/"'
                if target not in routes:
                    errors.append(f"{rel}: broken link {h}")
                    return mo.group(0)
                tr = routes[target]
            if frag and frag not in ids[tr]:
                errors.append(f"{rel}: broken anchor {h} (no id '{frag}' on {tr or 'home'})")
            return f'href="#/{tr}' + (f"#{frag}" if frag else "") + '"'

        body = href_re.sub(fix, body)
        # external links open in a new tab
        body = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener"', body)

        headings = [{"id": hid, "level": int(lv), "text": text_of(txt)}
                    for lv, hid, txt in re.findall(r'<h([23]) id="([^"]+)"[^>]*>(.*?)</h\1>', body, re.S)]
        (tmp_out / "pages" / f"{page_id(r)}.html").write_text(body)

        # reading time: prose at 200 wpm, code skimmed at twice that
        code_words = sum(len(text_of(c).split()) for c in re.findall(r"<pre.*?</pre>", body, re.S))
        prose = re.sub(r"<(script|style).*?</\1>", " ", re.sub(r"<pre.*?</pre>", " ", body, flags=re.S), flags=re.S)
        prose_words = len(text_of(prose).split())
        minutes = max(1, round((prose_words + 0.5 * code_words) / 200))

        meta[r].update({"id": page_id(r), "headings": headings, "minutes": minutes,
                        "hasMath": "arithmatex" in body, "hasMermaid": 'class="mermaid"' in body,
                        "hasScript": "<script" in body, "hasTasks": "task-list-item" in body})

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
        shutil.rmtree(tmp_out, ignore_errors=True)
        return 1

    # "Used in": concept page → the project steps that apply it (docs/curriculum.yml: used_in)
    for rel, uses in (cur.get("used_in") or {}).items():
        if rel not in routes:
            errors.append(f"curriculum.yml used_in: unknown page {rel}")
            continue
        out = []
        for u in uses:
            if u["project"] not in routes:
                errors.append(f"curriculum.yml used_in[{rel}]: unknown project page {u['project']}")
                continue
            pr = routes[u["project"]]
            out.append({"route": pr, "title": meta[pr]["title"], "note": u.get("note", "")})
        meta[routes[rel]]["usedIn"] = out
    if errors:
        print("\n".join(errors), file=sys.stderr)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return 1

    for rel, tag in (cur.get("tags") or {}).items():
        if rel in routes:
            meta[routes[rel]]["tag"] = tag
    redirects = {k: v for k, v in (cur.get("redirects") or {}).items() if v in meta}
    manifest = {"site": cur["site"], "tracks": tracks, "reference": reference, "order": order, "pages": meta, "redirects": redirects}
    (tmp_out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (tmp_out / "search.json").write_text(json.dumps(search, ensure_ascii=False))
    if final_out.exists():
        shutil.rmtree(final_out)
    tmp_out.rename(final_out)
    print(f"built {len(routes)} pages, {len(search)} search records → {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
