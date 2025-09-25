from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from .config import CognitiveModelConfig

logger = logging.getLogger(__name__)


@dataclass
class CognitiveInference:
    load: float
    confidence: float


class CognitiveStateModel:
    def __init__(self, config: CognitiveModelConfig) -> None:
        self._config = config
        self._model = None
        self._scaler = None
        self._feature_keys: Optional[list[str]] = None
        if config.model_path:
            self._load_model(config.model_path)

    def _load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        artifact = joblib.load(path)
        self._model = artifact.get("model")
        self._scaler = artifact.get("scaler")
        self._feature_keys = artifact.get("feature_keys")
        logger.info("Loaded cognitive state model from %s", model_path)

    def predict(self, features: Dict[str, float]) -> CognitiveInference:
        if self._feature_keys is None:
            self._feature_keys = sorted(features.keys())
        vector = np.array([features[k] for k in self._feature_keys], dtype=np.float32)
        if self._model is not None:
            if self._scaler is not None:
                vector = self._scaler.transform([vector])[0]
            prob = float(self._model.predict_proba([vector])[0, 1])
            confidence = self._confidence_from_prob(prob)
            return CognitiveInference(load=prob, confidence=confidence)
        return self._heuristic_prediction(features)

    def _heuristic_prediction(self, features: Dict[str, float]) -> CognitiveInference:
        engagement = features.get("engagement_index", 0.5)
        theta_beta = features.get("theta_beta_ratio", 1.0)
        entropy = features.get("spectral_entropy", 0.5)
        rel_alpha = features.get("rel_alpha", 0.2)
        rel_beta = features.get("rel_beta", 0.2)
        load = 0.5 * engagement + 0.3 * (1 - rel_alpha) + 0.2 * (theta_beta)
        load = float(np.clip(load / 2.0, 0.0, 1.0))
        confidence = float(np.clip(0.6 + 0.4 * (entropy - 0.3), 0.1, 0.95))
        if theta_beta > 2.5:
            confidence = min(confidence, 0.7)
        return CognitiveInference(load=load, confidence=confidence)

    def _confidence_from_prob(self, prob: float) -> float:
        return float(np.clip(abs(prob - 0.5) * 2, 0.05, 1.0))

    def smooth(self, inference: CognitiveInference, previous: Optional[CognitiveInference]) -> CognitiveInference:
        if previous is None:
            return inference
        alpha = self._config.smoothing_alpha
        beta = self._config.smoothing_beta
        load = alpha * inference.load + (1 - alpha) * previous.load
        confidence = beta * inference.confidence + (1 - beta) * previous.confidence
        return CognitiveInference(load=load, confidence=confidence)
