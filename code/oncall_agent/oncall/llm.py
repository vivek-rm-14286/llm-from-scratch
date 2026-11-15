"""The model call, for any OpenAI-compatible /chat/completions endpoint (standard library only).

    export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b      # Ollama
    export LLM_BASE_URL=http://localhost:8000/v1  LLM_MODEL=Qwen/Qwen2.5-7B-Instruct   # vLLM

Every client in this project (real or fake) has one method:
    chat(messages, tools) -> {"message": {...assistant message...},
                              "usage": {"prompt_tokens": int, "completion_tokens": int}}
"""
from __future__ import annotations

import json
import os
import urllib.request


def estimate_tokens(obj) -> int:
    """Rough count for endpoints (and fakes) that report no usage: about 4 characters per token."""
    return max(1, len(obj if isinstance(obj, str) else json.dumps(obj)) // 4)


class OpenAIChat:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, timeout: float = 120.0):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "not-needed")
        self.temperature, self.timeout = temperature, timeout

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        message = {k: v for k, v in out["choices"][0]["message"].items() if v is not None}
        usage = out.get("usage") or {}
        return {"message": message,
                "usage": {"prompt_tokens": usage.get("prompt_tokens") or estimate_tokens(messages),
                          "completion_tokens": usage.get("completion_tokens") or estimate_tokens(message)}}
