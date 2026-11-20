"""
OpenAI SDK runtime — dùng cho:

  Blue Team → OpenRouter liquid/lfm-3-1.7b:free (create_blue_pair)
  Red Team  → OpenAI gpt-4o-mini (create_openai_pair) khi RED_TEAM_PROVIDER=openai

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import (
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    blue_client_kwargs,
    red_openai_client_kwargs,
)

LOCAL_BLUE_MODEL = "lfm3-1.7b-local"
LOCAL_BLUE_BASE_URL = "http://127.0.0.1:18083/v1"


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class _MockInvocationContext:
    user_id: str = "student"


@dataclass(frozen=True)
class ChatResult:
    text: str
    blocked_by: str | None = None


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float = 0.4
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)
    response_generator: Callable[[str], str] | None = None

    def _client(self):
        from openai import OpenAI

        return OpenAI(**(self.client_kwargs or {}))

    async def chat(self, agent: OpenAIAgent, user_message: str) -> str:
        return (await self.chat_with_decision(agent, user_message)).text

    async def chat_with_decision(
        self, agent: OpenAIAgent, user_message: str, *, user_id: str = "student",
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Return the final reply and which input layer stopped the request."""
        blocked = await self.check_input(user_message, user_id=user_id)
        if blocked is not None:
            return blocked

        if self.response_generator is None:
            text = self._generate_text(agent, user_message, max_tokens=max_tokens)
        else:
            text = self.response_generator(user_message)

        for hook in self.output_hooks:
            text = hook(text)

        return ChatResult(await self._run_output_plugins(text))

    def _generate_text(
        self, agent: OpenAIAgent, user_message: str, *, max_tokens: int | None
    ) -> str:
        client = self._client()
        generation_options = {"max_tokens": max_tokens} if max_tokens is not None else {}
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": agent.instruction},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            **generation_options,
        )
        return (completion.choices[0].message.content or "").strip()

    async def check_input(self, user_message: str, *, user_id: str = "student") -> ChatResult | None:
        """Run pre-model checks and return the first blocking decision."""
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                return ChatResult(blocked, "input_hook")

        if not self.plugins:
            return None
        try:
            from google.genai import types
        except ImportError:
            return None

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        ctx = _MockInvocationContext(user_id=user_id)
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            try:
                result = await cb(
                    invocation_context=ctx, user_message=user_content
                )
            except TypeError:
                result = cb(invocation_context=ctx, user_message=user_content)
            if result is None:
                continue
            return ChatResult(_content_to_text(result), getattr(plugin, "name", "input_plugin"))
        return None

    async def _run_output_plugins(self, text: str) -> str:
        if not self.plugins or not text:
            return text
        try:
            from google.genai import types
        except ImportError:
            return text

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        class _Ctx:
            pass

        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            try:
                out = await cb(callback_context=_Ctx(), llm_response=llm_response)
            except TypeError:
                out = cb(callback_context=_Ctx(), llm_response=llm_response)
            if out is not None and getattr(out, "content", None) is not None:
                llm_response = out
        return _content_to_text(llm_response.content) or text


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    local: bool = False,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Build Blue with OpenRouter or the optional local LFM3-1.7B server."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=LOCAL_BLUE_MODEL if local else get_blue_model(),
        provider="llama.cpp" if local else get_blue_provider(),
        client_kwargs=(
            {"base_url": LOCAL_BLUE_BASE_URL, "api_key": "local"}
            if local else blue_client_kwargs()
        ),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Red Team OpenAI path (default = soft model; advance may pass harder)."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=get_red_provider(),
        client_kwargs=red_openai_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )
