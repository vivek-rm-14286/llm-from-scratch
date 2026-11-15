"""Pydantic schema + robust parsing for triage outputs (the guardrail layer)."""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from prompts import CATEGORIES, ESCALATION_TARGETS, SEVERITIES


class Triage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal[SEVERITIES]  # type: ignore[valid-type]
    category: Literal[CATEGORIES]  # type: ignore[valid-type]
    affected_service: str = Field(min_length=1, max_length=100)
    probable_root_cause: str = Field(min_length=5, max_length=600)
    recommended_next_steps: list[str] = Field(min_length=1, max_length=8)
    escalate_to: Literal[ESCALATION_TARGETS]  # type: ignore[valid-type]
    confidence: float = Field(ge=0.0, le=1.0)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> str:
    """Strip code fences / chatter and return the outermost {...} block."""
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return text[start : end + 1]


def parse_triage(text: str) -> tuple[Optional[Triage], Optional[str]]:
    """Returns (triage, None) on success or (None, error_message) on failure."""
    try:
        data = json.loads(extract_json(text))
        return Triage.model_validate(data), None
    except ValidationError as e:  # must come first: ValidationError subclasses ValueError
        return None, f"schema violation: {e.errors(include_url=False)}"
    except ValueError as e:  # includes json.JSONDecodeError
        return None, f"invalid JSON: {e}"


def json_schema() -> dict:
    """JSON schema for constrained decoding (vLLM / Ollama structured outputs)."""
    return Triage.model_json_schema()
