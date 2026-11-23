"""Qwen LLM implementation.

Qwen uses an OpenAI-compatible API over DashScope's compatible-mode endpoint.
This implementation reuses the OpenAI-compatible request/response contract while
adding provider-specific defaults and validation.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class QwenLLMError(RuntimeError):
    """Raised when Qwen API call fails."""


class QwenLLM(BaseLLM):
    """Qwen text generation provider via DashScope OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens

        self.api_key = (
            api_key
            or getattr(settings.llm, "api_key", None)
            or os.environ.get("QWEN_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Qwen API key not provided. Set settings.llm.api_key, "
                "QWEN_API_KEY environment variable, or pass api_key parameter."
            )

        self.base_url = (
            base_url
            or getattr(settings.llm, "base_url", None)
            or self.DEFAULT_BASE_URL
        )
        self._extra_config = kwargs

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.validate_messages(messages)

        import httpx

        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            raise QwenLLMError(f"Qwen API call failed: {e}") from e

        try:
            content = data["choices"][0]["message"]["content"]
            return ChatResponse(
                content=content,
                model=data.get("model", model),
                usage=data.get("usage"),
                raw_response=data,
            )
        except (KeyError, IndexError, TypeError) as e:
            raise QwenLLMError(f"Qwen API response format unexpected: {e}") from e
