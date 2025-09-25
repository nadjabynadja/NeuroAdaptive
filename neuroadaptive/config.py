from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EEGChannelConfig:
    board_id: int = 0
    sampling_rate: int = 256
    eeg_channels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    aux_channels: List[int] = field(default_factory=list)
    notch_freq: float = 50.0
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    window_size_seconds: float = 2.0
    step_size_seconds: float = 0.5


@dataclass
class PersonalizationConfig:
    baseline_duration_seconds: int = 120
    recalibration_interval_minutes: int = 30
    adaptation_rate: float = 0.02
    min_directive_interval_seconds: float = 2.0
    high_load_threshold: float = 0.7
    low_load_threshold: float = 0.3


@dataclass
class CognitiveModelConfig:
    model_path: Optional[str] = None
    smoothing_alpha: float = 0.6
    smoothing_beta: float = 0.2
    low_confidence_threshold: float = 0.35


@dataclass
class DirectiveConfig:
    verbosity_map: Dict[str, str] = field(
        default_factory=lambda: {
            "low": "Provide concise bullet summaries.",
            "medium": "Provide balanced detail with short paragraphs.",
            "high": "Provide thorough explanations with step-by-step reasoning.",
        }
    )
    tone_map: Dict[str, str] = field(
        default_factory=lambda: {
            "default": "Maintain a calm, encouraging tone.",
            "high_load": "Use reassuring, supportive language and offer breaks.",
            "engaged": "Match the user''s energy with collaborative language.",
        }
    )


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 500
    endpoint: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class SystemConfig:
    eeg: EEGChannelConfig = field(default_factory=EEGChannelConfig)
    personalization: PersonalizationConfig = field(default_factory=PersonalizationConfig)
    cognitive_model: CognitiveModelConfig = field(default_factory=CognitiveModelConfig)
    directives: DirectiveConfig = field(default_factory=DirectiveConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
