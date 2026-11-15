"""Pick which implementation of the learner-written modules to use: solution/ or starter/."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

MODULES = ("guardrails", "approval", "agent", "evaluate", "mcp_server")


def load_impl(name: str = "solution") -> SimpleNamespace:
    if name not in ("solution", "starter"):
        raise ValueError("impl must be 'solution' or 'starter'")
    return SimpleNamespace(name=name, **{m: importlib.import_module(f"{name}.{m}") for m in MODULES})
