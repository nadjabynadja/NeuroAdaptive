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
        buffer = self._rng.normal(0, 1e-6, size=(len(self._config.eeg_channels), window_samples))
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
        base_freqs = np.array([10.0, 20.0, 6.0, 12.0])[: len(self._config.eeg_channels)]
        phases = self._rng.uniform(0, 2 * np.pi, size=len(self._config.eeg_channels))
        oscillations = np.sin(2 * np.pi * base_freqs * t + phases) * 10e-6
        noise = self._rng.normal(0.0, 2e-6, size=len(self._config.eeg_channels))
        return oscillations + noise


def historical_frame_loader(frames: Iterable[np.ndarray]) -> AsyncGenerator[EEGFrame, None]:
    async def generator() -> AsyncGenerator[EEGFrame, None]:
        for eeg in frames:
            timestamps = np.linspace(0, eeg.shape[1] / 256.0, eeg.shape[1])
            yield EEGFrame(timestamps=timestamps, eeg=eeg)
            await asyncio.sleep(0)
    return generator()
