"""A transparent programmatic judge: six boolean checks, a score, and pair selection."""
import re

from constraints import satisfies, words

CHECKS = ("finished", "not_empty", "clean", "no_repeat", "on_topic", "constraint")
_LEAKS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
_ROLE_LINE = re.compile(r"^\s*(user|assistant|system)\s*:?\s*$", re.IGNORECASE)
STOPWORDS = frozenset("""about above after again also answer because been before being
between both could does doing down during each explain from further give have having here
into just list make many more most much must name only other over please same should some
such tell than that their them then there these they this those through under until very
what when where which while will with would your""".split())


def content_words(text):
    """Lowercased alphabetic words of 4+ letters that are not stopwords."""
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


def has_repeated_ngram(text, n=4):
    """True if any sequence of n consecutive words occurs twice (case-insensitive)."""
    toks = [w.lower() for w in words(text)]
    seen = set()
    for i in range(len(toks) - n + 1):
        gram = tuple(toks[i : i + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def check_response(question, response, constraint, finished):
    """Run the rubric. Returns a dict with one bool per name in CHECKS.

    question:   the user's question WITHOUT the constraint instruction
    response:   the generated text (special tokens not stripped)
    constraint: a constraint dict (see constraints.py)
    finished:   True if generation stopped by emitting <|im_end|> before the token limit
    """
    text = response.strip()
    q_words = content_words(question)
    return {
        "finished": bool(finished),
        "not_empty": len(words(text)) >= 3,
        "clean": not any(m in text for m in _LEAKS)
                 and not any(_ROLE_LINE.match(ln) for ln in text.splitlines()),
        "no_repeat": not has_repeated_ngram(text, 4),
        "on_topic": not q_words or bool(q_words & content_words(text)),
        "constraint": bool(text) and satisfies(text, constraint),
    }


def judge(question, response, constraint, finished):
    """Returns {"checks": {...}, "score": passed-check count 0..6, "passed": all six}."""
    checks = check_response(question, response, constraint, finished)
    score = sum(checks.values())
    return {"checks": checks, "score": score, "passed": score == len(CHECKS)}


def pick_pair(scores, min_gap=1):
    """Choose (chosen_index, rejected_index) from one prompt's candidate scores.

    chosen = highest score, rejected = lowest score; ties go to the lowest index, so
    length never breaks a tie. Returns None if best - worst < min_gap.
    """
    if not scores:
        return None
    best = max(range(len(scores)), key=lambda i: (scores[i], -i))
    worst = min(range(len(scores)), key=lambda i: (scores[i], i))
    if scores[best] - scores[worst] < min_gap:
        return None
    return best, worst
