from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional dependency
    AsyncOpenAI = None  # type: ignore

from .config import LLMConfig
from .directives import AdaptationDirective


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ConversationContext:
    messages: List[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def as_list(self) -> List[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]


class LLMOrchestrator:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._directive: Optional[AdaptationDirective] = None
        self._conversation = ConversationContext()
        self._client = None
        if config.provider == "openai" and AsyncOpenAI is not None:
            api_key = os.getenv(config.api_key_env)
            if api_key:
                self._client = AsyncOpenAI(api_key=api_key, base_url=config.endpoint)

    def set_directive(self, directive: AdaptationDirective) -> None:
        self._directive = directive

    def add_user_message(self, content: str) -> None:
        self._conversation.add("user", content)

    async def generate_response(self) -> str:
        if self._directive is None:
            raise RuntimeError("Directive not set before generating response")
        system_msg = self._directive.system_instruction
        metadata_desc = "/".join(f"{k}={v}" for k, v in self._directive.metadata.items())
        system_msg = f"{system_msg}\nCurrent load context: {metadata_desc}."
        self._conversation.messages.insert(0, Message(role="system", content=system_msg))
        try:
            if self._client:
                response = await self._client.chat.completions.create(
                    model=self._config.model,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    messages=self._conversation.as_list(),
                )
                content = response.choices[0].message.content or ""
            else:
                content = self._fallback_response()
        finally:
            self._conversation.messages.pop(0)
        self._conversation.add("assistant", content)
        return content

    def _fallback_response(self) -> str:
        history = "\n".join(f"{m.role}: {m.content}" for m in self._conversation.messages[-4:])
        directive = self._directive.system_instruction if self._directive else ""
        return (
            "[stubbed-response] Following directive: "
            + directive
            + " | Recent messages: "
            + history
        )

    def reset(self) -> None:
        self._conversation = ConversationContext()
        self._directive = None
