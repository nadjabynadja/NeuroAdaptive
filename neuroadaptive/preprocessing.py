from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .config import EEGChannelConfig

logger = logging.getLogger(__name__)


def design_bandpass(low: float, high: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    nyquist = fs / 2.0
    low_norm = max(low / nyquist, 1e-4)
    high_norm = min(high / nyquist, 0.99)
    if low_norm >= high_norm:
        raise ValueError("Invalid bandpass frequencies")
    return butter(order, [low_norm, high_norm], btype="band")


def design_notch(freq: float, fs: float, quality: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    w0 = freq / (fs / 2.0)
    if not 0 < w0 < 1:
        raise ValueError("Notch frequency must be between 0 and Nyquist")
    return iirnotch(w0, quality)


def apply_filters(samples: np.ndarray, config: EEGChannelConfig) -> np.ndarray:
    b_band, a_band = design_bandpass(config.bandpass_low, config.bandpass_high, config.sampling_rate)
    bandpassed = filtfilt(b_band, a_band, samples, axis=-1)
    b_notch, a_notch = design_notch(config.notch_freq, config.sampling_rate)
    filtered = filtfilt(b_notch, a_notch, bandpassed, axis=-1)
    return filtered


def suppress_artifacts(samples: np.ndarray, z_threshold: float = 5.0) -> np.ndarray:
    z_scores = (samples - samples.mean(axis=-1, keepdims=True)) / (samples.std(axis=-1, keepdims=True) + 1e-8)
    mask = np.abs(z_scores) > z_threshold
    cleaned = samples.copy()
    if np.any(mask):
        medians = np.median(samples, axis=-1, keepdims=True)
        cleaned[mask] = medians.repeat(samples.shape[-1], axis=-1)[mask]
        logger.debug("Artifact suppression replaced %s samples", mask.sum())
    return cleaned


def preprocess_frame(eeg: np.ndarray, config: EEGChannelConfig) -> np.ndarray:
    filtered = apply_filters(eeg, config)
    cleaned = suppress_artifacts(filtered)
    return cleaned
