from __future__ import annotations

from dataclasses import dataclass

from .config import DirectiveConfig
from .personalization import DirectiveContext


@dataclass
class AdaptationDirective:
    verbosity_label: str
    tone_label: str
    system_instruction: str
    metadata: dict


class DirectiveMapper:
    def __init__(self, config: DirectiveConfig) -> None:
        self._config = config

    def map(self, context: DirectiveContext) -> AdaptationDirective:
        verbosity_label = self._select_verbosity(context)
        tone_label = self._select_tone(context)
        instruction = self._build_instruction(verbosity_label, tone_label, context)
        metadata = {
            "load": context.normalized_load,
            "load_level": context.load_level,
            "confidence": context.confidence,
            "trend": context.trend,
            "suppressed": context.suppress_adaptation,
        }
        return AdaptationDirective(
            verbosity_label=verbosity_label,
            tone_label=tone_label,
            system_instruction=instruction,
            metadata=metadata,
        )

    def _select_verbosity(self, context: DirectiveContext) -> str:
        if context.suppress_adaptation:
            return "medium"
        return {
            "high": "low",
            "medium": "medium",
            "low": "high",
        }[context.load_level]

    def _select_tone(self, context: DirectiveContext) -> str:
        if context.suppress_adaptation:
            return "default"
        if context.load_level == "high":
            return "high_load"
        if context.trend == "rising":
            return "high_load"
        return "engaged"

    def _build_instruction(self, verbosity: str, tone: str, context: DirectiveContext) -> str:
        verbosity_instruction = self._config.verbosity_map.get(verbosity, "Provide balanced detail.")
        tone_instruction = self._config.tone_map.get(tone, "Maintain a calm, encouraging tone.")
        gating = "Only adjust if confidence is high." if context.suppress_adaptation else ""
        return " ".join(part for part in [verbosity_instruction, tone_instruction, gating] if part)
