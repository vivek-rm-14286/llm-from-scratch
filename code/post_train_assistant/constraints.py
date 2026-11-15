"""Checkable answer constraints: text helpers shared by data preparation and the judge.

A constraint is a small dict:
    {"kind": "max_words", "n": 20}      "Answer in at most 20 words."
    {"kind": "one_sentence"}            "Answer in one sentence."
    {"kind": "numbered_list", "n": 3}   "Answer with a numbered list of exactly 3 items."
"""
import re

WORD_LIMITS = (20, 40)
LIST_SIZES = (3, 4, 5)
_ITEM = re.compile(r"^\s*\d+[.)]\s+\S")


def instruction_for(constraint):
    kind = constraint["kind"]
    if kind == "max_words":
        return f"Answer in at most {constraint['n']} words."
    if kind == "one_sentence":
        return "Answer in one sentence."
    if kind == "numbered_list":
        return f"Answer with a numbered list of exactly {constraint['n']} items."
    raise ValueError(f"unknown constraint kind: {kind}")


def words(text):
    """Whitespace-separated tokens."""
    return text.split()


def sentences(text):
    """Split on ., ! or ? followed by whitespace or the end of the text.
    Deliberately simple: 'e.g. this' counts as two sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def list_items(text):
    """Lines that look like '1. item' or '2) item'."""
    return [ln for ln in text.splitlines() if _ITEM.match(ln)]


def is_pure_list(text):
    """True if every non-empty line is a numbered item."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and all(_ITEM.match(ln) for ln in lines)


def satisfies(text, constraint):
    """Does `text` meet the constraint? Pure string logic, no model."""
    kind = constraint["kind"]
    if kind == "max_words":
        return 0 < len(words(text)) <= constraint["n"]
    if kind == "one_sentence":
        return len(sentences(text)) == 1 and "\n" not in text.strip()
    if kind == "numbered_list":
        return is_pure_list(text) and len(list_items(text)) == constraint["n"]
    raise ValueError(f"unknown constraint kind: {kind}")


def constraints_met_by(answer):
    """All constraints from the fixed menu that an existing answer already satisfies.
    Used for hindsight relabeling of SFT data: the tightest word limit, one sentence,
    and a list size, when they apply."""
    out = []
    n = len(words(answer))
    for limit in WORD_LIMITS:
        if 0 < n <= limit:
            out.append({"kind": "max_words", "n": limit})
            break
    if satisfies(answer, {"kind": "one_sentence"}):
        out.append({"kind": "one_sentence"})
    for k in LIST_SIZES:
        if satisfies(answer, {"kind": "numbered_list", "n": k}):
            out.append({"kind": "numbered_list", "n": k})
    return out
