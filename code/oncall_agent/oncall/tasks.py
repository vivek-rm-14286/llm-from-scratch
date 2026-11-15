"""An incident as the agent receives it: a kind, the caller's access level and the raw text."""
from __future__ import annotations

import re
from dataclasses import dataclass

KINDS = ("alert", "ticket", "question")
_HEADER = re.compile(r"Incident kind: (\w+)\nCaller access: (\w+)\n--- incident ---\n(.*)\n--- end ---", re.S)


@dataclass
class Task:
    kind: str            # alert (from the monitoring system), ticket or question (written by a person)
    access: str          # public | internal | restricted: what the caller may read
    text: str
    id: str = ""


def render_task(task: Task, body: str | None = None) -> str:
    """The user message. Pass `body` to send a sanitized version of task.text."""
    return (f"Incident kind: {task.kind}\nCaller access: {task.access}\n"
            f"--- incident ---\n{task.text if body is None else body}\n--- end ---")


def parse_task(user_content: str) -> tuple[str, str, str]:
    m = _HEADER.search(user_content)
    return (m.group(1), m.group(2), m.group(3)) if m else ("alert", "internal", user_content)
