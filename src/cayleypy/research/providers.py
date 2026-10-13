"""Thread-safe clients for live research models."""

import json
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass
class Completion:
    data: dict
    model: str
    usage: dict = field(default_factory=dict)


class Provider(Protocol):
    def complete(self, role: str, system: str, payload: dict) -> Completion: ...


class ProviderError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("LLM endpoint redirects are disabled; configure the final URL")


class ChatCompletionsProvider:
    """OpenAI-compatible JSON chat API; model selection is explicit, keys stay in env."""

    def __init__(self, *, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 120,
                 max_completion_tokens: int = 4096, role_models: dict | None = None,
                 reasoning_effort: str | None = None, json_mode: bool = True):
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("LLM URL requires HTTPS, except local HTTP servers")
        if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
            raise ValueError("LLM base URL must not contain credentials, query, or fragment")
        if not model or not isinstance(model, str):
            raise ValueError("configure an explicit LLM model")
        if timeout <= 0 or not 1 <= max_completion_tokens <= 100000:
            raise ValueError("invalid LLM timeout or completion token budget")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key_env, self.timeout = api_key_env, timeout
        self.max_completion_tokens, self.role_models = max_completion_tokens, role_models or {}
        self.reasoning_effort, self.json_mode = reasoning_effort, json_mode

    def complete(self, role: str, system: str, payload: dict) -> Completion:
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if self.api_key_env and not key:
            raise ProviderError(f"Set {self.api_key_env} to use the configured LLM endpoint")
        model = self.role_models.get(role, self.model)
        body = {"model": model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "max_completion_tokens": self.max_completion_tokens}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
        try:
            with build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ProviderError("LLM response exceeds the 4 MiB limit")
            result = json.loads(raw)
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ProviderError(f"LLM did not finish normally: {choice.get('finish_reason')}")
            message = choice["message"]
            if message.get("refusal"):
                raise ProviderError("LLM declined the request")
            data = json.loads(message["content"])
            if not isinstance(data, dict):
                raise ProviderError("LLM must return a JSON object")
            return Completion(data, result.get("model", model), result.get("usage", {}))
        except HTTPError as exc:
            # Do not echo server response bodies, request headers, or credentials.
            raise ProviderError(f"LLM HTTP {exc.code}; check model, endpoint, quota and credentials") from None
        except (URLError, TimeoutError) as exc:
            raise ProviderError(f"LLM transport failed ({type(exc).__name__})") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("LLM response did not match the required JSON contract") from None
