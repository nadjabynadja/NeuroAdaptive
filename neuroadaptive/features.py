from __future__ import annotations

from math import log2
from typing import Dict, Tuple

import numpy as np


FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _power_spectral_density(samples: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    n = samples.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(samples, axis=-1)
    psd = (np.abs(fft_vals) ** 2) / (fs * n)
    return freqs, psd


def band_powers(samples: np.ndarray, fs: float) -> Dict[str, float]:
    freqs, psd = _power_spectral_density(samples, fs)
    band_power: Dict[str, float] = {}
    for band, (low, high) in FREQUENCY_BANDS.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power[band] = psd[:, idx].sum(axis=-1).mean()
    return band_power


def relative_band_powers(samples: np.ndarray, fs: float) -> Dict[str, float]:
    abs_powers = band_powers(samples, fs)
    total_power = sum(abs_powers.values()) + 1e-12
    return {band: power / total_power for band, power in abs_powers.items()}


def theta_beta_ratio(samples: np.ndarray, fs: float) -> float:
    abs_powers = band_powers(samples, fs)
    return abs_powers["theta"] / (abs_powers["beta"] + 1e-12)


def engagement_index(samples: np.ndarray, fs: float) -> float:
    abs_powers = band_powers(samples, fs)
    return (abs_powers["beta"] + abs_powers["gamma"]) / (abs_powers["alpha"] + abs_powers["theta"] + 1e-12)


def spectral_entropy(samples: np.ndarray, fs: float) -> float:
    _, psd = _power_spectral_density(samples, fs)
    psd_norm = psd / (psd.sum(axis=-1, keepdims=True) + 1e-12)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=-1).mean()
    return entropy / log2(psd.shape[-1])


def frontal_asymmetry(samples: np.ndarray) -> float:
    if samples.shape[0] < 2:
        return 0.0
    left, right = samples[0], samples[1]
    return float(np.log(np.var(right) + 1e-12) - np.log(np.var(left) + 1e-12))


def extract_features(samples: np.ndarray, fs: float) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    feats.update({f"rel_{band}": value for band, value in relative_band_powers(samples, fs).items()})
    feats["theta_beta_ratio"] = theta_beta_ratio(samples, fs)
    feats["engagement_index"] = engagement_index(samples, fs)
    feats["spectral_entropy"] = spectral_entropy(samples, fs)
    feats["frontal_asymmetry"] = frontal_asymmetry(samples)
    feats["mean_amplitude"] = float(np.mean(np.abs(samples)))
    feats["std_amplitude"] = float(np.std(samples))
    return feats


def feature_vector(feature_dict: Dict[str, float]) -> np.ndarray:
    keys = sorted(feature_dict.keys())
    return np.array([feature_dict[k] for k in keys], dtype=np.float32)
