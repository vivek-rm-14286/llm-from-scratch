"""Guardrails around generation: an input check, a scan of retrieved chunks, and an output filter.

Three cheap, deterministic layers. None of them is a security boundary on its own: pattern
matching is bypassed by rephrasing, which is why `redteam.py` measures what still gets through.
The hard guarantees in this project come from elsewhere: the access filter inside retrieval, and
the fact that the copilot can only produce text (no tools, no side effects).
"""
from __future__ import annotations

import re

import settings

# Phrases that address the model rather than describe the product. Kept short and specific so
# ordinary support text does not match (measured on the clean corpus by redteam.py).
INJECTION_PATTERNS = {
    "override": re.compile(r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your|system)\b[^.\n]{0,20}\b(instructions?|rules?|prompts?|guidelines?)", re.I),
    "role_marker": re.compile(r"(^|\n)\s*(system|assistant|developer)\s*(note|message|override)?\s*:", re.I),
    "addresses_model": re.compile(r"\b(ai|llm|language model|chatbot|copilot)\b[^.\n]{0,30}\b(must|should|shall|has to|needs to|is required to)\b", re.I),
    "prompt_leak": re.compile(r"\b(system prompt|initial prompt|your (hidden |original )?instructions)\b", re.I),
}

URL = re.compile(r"https?://[^\s<>()\[\]\"'`]+", re.I)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def scan(text: str) -> list[str]:
    """Names of the injection patterns found in a piece of text."""
    return [name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(text)]


def check_question(question: str) -> list[str]:
    return scan(question)


def filter_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split retrieved chunks into (kept, quarantined). A quarantined chunk never reaches the prompt;
    log its id so someone can look at the document it came from."""
    kept, dropped = [], []
    for c in chunks:
        hits = scan(c["text"])
        (dropped if hits else kept).append({**c, "guard_hits": hits} if hits else c)
    return kept, dropped


def _host_allowed(host: str) -> bool:
    host = host.lower().split("@")[-1].split(":")[0].rstrip(".")
    return any(host == d or host.endswith("." + d) for d in settings.ALLOWED_LINK_DOMAINS)


def sanitize_answer(text: str, system_prompt: str = "") -> tuple[str, list[str]]:
    """Remove what a hijacked answer would use to do damage: images (their URL is fetched by the
    client, a data-exfiltration channel), links and email addresses outside the allowlist, and long
    verbatim spans of the system prompt."""
    findings = []
    if MD_IMAGE.search(text):
        text = MD_IMAGE.sub("[image removed]", text)
        findings.append("image")

    def link(m: re.Match) -> str:
        host = re.sub(r"^https?://", "", m.group(0), flags=re.I).split("/")[0]
        if _host_allowed(host):
            return m.group(0)
        findings.append(f"link:{host}")
        return "[link removed]"

    def email(m: re.Match) -> str:
        if _host_allowed(m.group(1)):
            return m.group(0)
        findings.append(f"email:{m.group(1)}")
        return "[address removed]"

    text = EMAIL.sub(email, URL.sub(link, text))
    for line in system_prompt.splitlines():
        line = line.strip()
        if len(line) >= 40 and line.lower() in text.lower():
            findings.append("system_prompt_leak")
            return "", findings  # caller replaces the whole answer
    return text, findings
