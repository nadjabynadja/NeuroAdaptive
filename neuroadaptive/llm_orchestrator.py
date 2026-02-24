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
        """Return a directive-aware stub response that demonstrates the adaptation.

        Adapts verbosity and tone to the active directive so the demo audience
        can see the injection working even without a live LLM API key.
        """
        if self._directive is None:
            return "[NEUROADAPTIVE STUB] No directive set."

        meta = self._directive.metadata
        verb = self._directive.verbosity_label
        tone = self._directive.tone_label
        load_pct = int(float(meta.get("load", 0.5)) * 100)
        level = str(meta.get("load_level", "medium"))
        conf_pct = int(float(meta.get("confidence", 0.0)) * 100)

        user_msgs = [m for m in self._conversation.messages if m.role == "user"]
        topic = user_msgs[-1].content[:80] if user_msgs else "your question"

        header = (
            f"[NEUROADAPTIVE STUB | load={level} ({load_pct}%) | "
            f"verbosity={verb} | tone={tone} | conf={conf_pct}%]\n"
        )

        if verb == "low":
            # High cognitive load → terse bullets, reassuring language
            body = (
                f"Key points on: '{topic[:60]}'\n\n"
                f"• Directive active: {self._directive.system_instruction}\n"
                f"• Response compressed — elevated cognitive load detected.\n"
                f"• Core answer: this stub demonstrates high-load brevity.\n\n"
                f"Take a moment; we can expand any point when you're ready."
            )
        elif verb == "high":
            # Low cognitive load → verbose, step-by-step, exploratory framing
            body = (
                f"Directive active: {self._directive.system_instruction}\n\n"
                f"Your EEG profile shows low cognitive load (alpha-dominant, "
                f"low theta-beta ratio) — the system switches to high-verbosity mode "
                f"and provides step-by-step reasoning.\n\n"
                f"On your question — '{topic[:70]}' —\n\n"
                f"Step 1: Context.  The neuroadaptive loop infers that you have "
                f"available working memory and are in an exploratory cognitive state.\n\n"
                f"Step 2: Mechanism.  The directive above is prepended to the LLM "
                f"system prompt before every API call, shaping response depth and tone "
                f"in real time without any explicit user instruction.\n\n"
                f"Step 3: Why it matters.  This stub replaces the live LLM call; in "
                f"production the same directive would instruct the model to deliver a "
                f"detailed, example-rich answer matched to your current capacity."
            )
        else:
            # Medium load → balanced paragraph, collaborative tone
            body = (
                f"Directive active: {self._directive.system_instruction}\n\n"
                f"Responding to '{topic[:70]}' at balanced verbosity "
                f"(moderate cognitive load, {level} level).\n\n"
                f"The EEG-derived context adjusts both response depth and tone. "
                f"In production this placeholder is replaced by an actual LLM call "
                f"with the directive injected as the system prompt — the model then "
                f"tailors verbosity, examples, and pacing to your real-time neural state."
            )

        return header + "\n" + body

    def reset(self) -> None:
        self._conversation = ConversationContext()
        self._directive = None
