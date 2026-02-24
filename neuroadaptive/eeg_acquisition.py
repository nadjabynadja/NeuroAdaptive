from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Iterable, Optional

import numpy as np

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
except ImportError:  # pragma: no cover - BrainFlow optional
    BoardShim = None  # type: ignore
    BrainFlowInputParams = object  # type: ignore

from .config import EEGChannelConfig

logger = logging.getLogger(__name__)


@dataclass
class EEGFrame:
    timestamps: np.ndarray
    eeg: np.ndarray
    aux: Optional[np.ndarray] = None


class EEGReader:
    """Async EEG producer.  In simulator mode it synthesises physiologically-
    grounded oscillations (fixed initial phases per channel, mode-switchable
    frequencies) so that downstream spectral features are meaningful.
    """

    # Each mode is a physiologically-motivated combination of dominant
    # frequencies.  The heuristic cognitive model responds to:
    #   engagement_index = (beta+gamma) / (alpha+theta)   – high → high load
    #   theta_beta_ratio = theta / beta                   – high → fatigue/overload
    #   rel_alpha                                          – high → relaxed
    #
    # The modes are tuned so that with a symmetric baseline (mean=0.5, std=0.25)
    # they produce clearly separated normalised load categories:
    #   relaxed   → raw ≈ 0.01  → normalised ≈ 0.13  → LOW
    #   overloaded → raw ≈ 1.0  → normalised ≈ 0.88  → HIGH
    #   fatigued  → raw ≈ 0.50  → normalised ≈ 0.50  → MEDIUM
    #   focused   → raw ≈ 1.0  → normalised ≈ 0.88   → HIGH (high-engagement variant)
    SIMULATOR_MODES: dict = {
        "relaxed": {
            # Pure alpha (8–13 Hz): resting, receptive state.
            # → Very high rel_alpha, near-zero engagement_index and theta_beta_ratio.
            # → raw_load ≈ 0.01  →  normalised LOW  →  verbosity=high, tone=engaged
            "freqs": [11.0, 12.0, 10.5, 11.5],
            "amps":  [15e-6, 12e-6, 14e-6, 13e-6],
            "noise":  1e-6,
        },
        "focused": {
            # Pure beta (13–30 Hz): sustained attention, high engagement.
            # → Very high engagement_index, near-zero rel_alpha.
            # → raw_load ≈ 1.0  →  normalised HIGH  →  verbosity=low, tone=high_load
            "freqs": [16.0, 20.0, 14.0, 18.0],
            "amps":  [ 9e-6, 10e-6,  8e-6, 11e-6],
            "noise":  3e-6,
        },
        "overloaded": {
            # Pure high-beta / gamma (18–28 Hz): cognitive overload.
            # → Extreme engagement_index, near-zero rel_alpha and theta_beta_ratio.
            # → raw_load ≈ 1.0  →  normalised HIGH  →  verbosity=low, tone=high_load
            "freqs": [20.0, 25.0, 22.0, 18.0],
            "amps":  [ 8e-6,  9e-6,  7e-6, 10e-6],
            "noise":  5e-6,
        },
        "fatigued": {
            # Mixed theta (5–7 Hz) + alpha (11 Hz) + low-beta (15 Hz): mental fatigue.
            # → Moderate engagement_index, moderate theta_beta_ratio, low rel_alpha.
            # → raw_load ≈ 0.50  →  normalised MEDIUM  →  verbosity=medium
            "freqs": [ 6.0, 15.0,  7.0, 11.0],
            "amps":  [15e-6, 12e-6, 14e-6,  8e-6],
            "noise":  2e-6,
        },
    }

    def __init__(
        self,
        config: EEGChannelConfig,
        params: Optional[BrainFlowInputParams] = None,
        use_simulator: bool = False,
        simulator_seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._params = params
        self._use_simulator = use_simulator or BoardShim is None
        self._rng = np.random.default_rng(simulator_seed)
        self._board: Optional[BoardShim] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._queue: asyncio.Queue[EEGFrame] = asyncio.Queue(maxsize=4)
        self._running = asyncio.Event()

        # Simulator state — fixed initial phases so each channel produces a
        # *coherent* oscillation (phase-continuous across samples) and the FFT
        # correctly resolves spectral peaks in the intended frequency bands.
        n = len(config.eeg_channels)
        self._channel_phases = self._rng.uniform(0, 2 * np.pi, size=n)
        self._base_freqs = np.array([10.0, 20.0, 6.0, 12.0])[:n]
        self._sim_amplitudes = np.full(n, 10e-6, dtype=float)
        self._sim_noise_std: float = 2e-6

    def set_mode(self, mode: str) -> None:
        """Switch simulator EEG spectral profile (no-op on real hardware).

        Calling this mid-session changes the dominant frequency mix so the
        cognitive model will infer a different load level within a few frames.

        Args:
            mode: One of "relaxed", "focused", "overloaded", "fatigued".
        """
        if not self._use_simulator:
            logger.warning("set_mode() ignored — running on real hardware.")
            return
        params = self.SIMULATOR_MODES.get(mode)
        if params is None:
            raise ValueError(
                f"Unknown simulator mode '{mode}'. "
                f"Choose from: {list(self.SIMULATOR_MODES)}"
            )
        n = len(self._config.eeg_channels)
        self._base_freqs = np.array(params["freqs"])[:n]
        self._sim_amplitudes = np.array(params["amps"])[:n]
        self._sim_noise_std = float(params["noise"])
        # Re-randomise phases on mode switch so the transition is clean.
        self._channel_phases = self._rng.uniform(0, 2 * np.pi, size=n)

    async def __aenter__(self) -> "EEGReader":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def start(self) -> None:
        if self._running.is_set():
            return
        if not self._use_simulator:
            if BoardShim is None:
                raise RuntimeError("BrainFlow is not available; enable simulator mode.")
            if self._params is None:
                raise ValueError("BrainFlowInputParams required when not using simulator.")
            BoardShim.enable_dev_board_logger()
            self._board = BoardShim(self._config.board_id, self._params)
            self._board.prepare_session()
            self._board.start_stream()
        self._running.set()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._producer())

    async def stop(self) -> None:
        self._running.clear()
        if self._task:
            await self._task
            self._task = None
        if self._board:
            try:
                self._board.stop_stream()
            except Exception:  # pragma: no cover - cleanup best effort
                logger.exception("Failed stopping BrainFlow stream")
            try:
                self._board.release_session()
            except Exception:  # pragma: no cover
                logger.exception("Failed releasing BrainFlow session")
            self._board = None
        while not self._queue.empty():
            self._queue.get_nowait()

    async def frames(self) -> AsyncGenerator[EEGFrame, None]:
        while self._running.is_set():
            frame = await self._queue.get()
            yield frame

    async def _producer(self) -> None:
        try:
            window_samples = int(self._config.sampling_rate * self._config.window_size_seconds)
            step_samples = max(1, int(self._config.sampling_rate * self._config.step_size_seconds))
            if self._use_simulator:
                await self._simulate(window_samples, step_samples)
            else:
                await self._stream_from_board(window_samples, step_samples)
        except Exception:  # pragma: no cover - ensure background errors surface
            logger.exception("EEG producer terminated unexpectedly")
        finally:
            self._running.clear()

    async def _stream_from_board(self, window_samples: int, step_samples: int) -> None:
        assert self._board is not None
        while self._running.is_set():
            await asyncio.sleep(self._config.step_size_seconds)
            data = self._board.get_board_data()
            if data.size == 0:
                continue
            eeg = data[self._config.eeg_channels, -window_samples:]
            aux = data[self._config.aux_channels, -window_samples:] if self._config.aux_channels else None
            timestamps = data[-1, -window_samples:]
            frame = EEGFrame(timestamps=timestamps, eeg=eeg, aux=aux)
            await self._queue.put(frame)

    async def _simulate(self, window_samples: int, step_samples: int) -> None:
        dt = 1.0 / self._config.sampling_rate
        t = 0.0
        n_ch = len(self._config.eeg_channels)
        buffer = self._rng.normal(0, 1e-6, size=(n_ch, window_samples))
        while self._running.is_set():
            await asyncio.sleep(self._config.step_size_seconds)
            samples = []
            timestamps = []
            for _ in range(step_samples):
                t += dt
                timestamps.append(t)
                sample = self._generate_sample(t)
                samples.append(sample)
            new_data = np.stack(samples, axis=1)
            buffer = np.concatenate([buffer[:, step_samples:], new_data], axis=1)
            frame = EEGFrame(timestamps=np.asarray(timestamps), eeg=buffer)
            await self._queue.put(frame)

    def _generate_sample(self, t: float) -> np.ndarray:
        """Generate one time-step of simulated EEG.

        Uses *persistent* per-channel phases so the oscillation is coherent
        across samples and the FFT correctly resolves spectral peaks.
        """
        oscillations = (
            np.sin(2 * np.pi * self._base_freqs * t + self._channel_phases)
            * self._sim_amplitudes
        )
        noise = self._rng.normal(0.0, self._sim_noise_std, size=len(self._config.eeg_channels))
        return oscillations + noise


def historical_frame_loader(frames: Iterable[np.ndarray]) -> AsyncGenerator[EEGFrame, None]:
    async def generator() -> AsyncGenerator[EEGFrame, None]:
        for eeg in frames:
            timestamps = np.linspace(0, eeg.shape[1] / 256.0, eeg.shape[1])
            yield EEGFrame(timestamps=timestamps, eeg=eeg)
            await asyncio.sleep(0)
    return generator()
