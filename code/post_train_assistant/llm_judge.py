"""Optional LLM judge through any OpenAI-compatible /chat/completions endpoint.

Set OPENAI_BASE_URL (for example http://localhost:11434/v1 for Ollama), OPENAI_API_KEY
(any string for local servers) and JUDGE_MODEL. Uses only the standard library.
The LLM score is added on top of the programmatic score as a tie-breaker; it never
overrides a failed programmatic check.
"""
import json
import os
import re
import urllib.request

PROMPT = """You grade answers from a small assistant. Rate the ANSWER to the QUESTION from 1 to 5
for correctness and helpfulness, given the INSTRUCTION it had to follow. Do not reward length.
Reply with only the number.

QUESTION: {question}
INSTRUCTION: {instruction}
ANSWER: {answer}"""


def llm_score(question, instruction, answer, timeout=60):
    """Returns a float in [0, 1], or None if the reply has no 1-5 digit."""
    body = json.dumps({
        "model": os.environ["JUDGE_MODEL"],
        "temperature": 0,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": PROMPT.format(
            question=question, instruction=instruction, answer=answer)}],
    }).encode()
    req = urllib.request.Request(
        os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none")})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = json.load(resp)["choices"][0]["message"]["content"]
    m = re.search(r"[1-5]", text)
    return (int(m.group()) - 1) / 4 if m else None
