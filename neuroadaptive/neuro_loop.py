from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Optional

from .cognitive_state import CognitiveInference, CognitiveStateModel
from .config import SystemConfig
from .directives import AdaptationDirective, DirectiveMapper
from .eeg_acquisition import EEGReader
from .features import extract_features
from .llm_orchestrator import LLMOrchestrator
from .personalization import DirectiveContext, PersonalizationManager
from .preprocessing import preprocess_frame


@dataclass
class LoopState:
    directive: Optional[AdaptationDirective] = None
    inference: Optional[CognitiveInference] = None


class NeuroadaptiveSession:
    def __init__(
        self,
        config: SystemConfig,
        reader: EEGReader,
        cognitive_model: CognitiveStateModel,
        personalizer: PersonalizationManager,
        directive_mapper: DirectiveMapper,
        orchestrator: LLMOrchestrator,
    ) -> None:
        self._config = config
        self._reader = reader
        self._model = cognitive_model
        self._personalizer = personalizer
        self._mapper = directive_mapper
        self._orchestrator = orchestrator
        self._state = LoopState()
        self._processing_task: Optional[asyncio.Task[None]] = None
        self._last_inference: Optional[CognitiveInference] = None

    async def start(self) -> None:
        await self._reader.start()
        self._processing_task = asyncio.create_task(self._process_frames())

    async def shutdown(self) -> None:
        if self._processing_task:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task
            self._processing_task = None
        await self._reader.stop()

    async def handle_user_message(self, message: str) -> str:
        directive = self._state.directive or self._default_directive()
        self._orchestrator.set_directive(directive)
        self._orchestrator.add_user_message(message)
        return await self._orchestrator.generate_response()

    async def _process_frames(self) -> None:
        async for frame in self._reader.frames():
            preprocessed = preprocess_frame(frame.eeg, self._config.eeg)
            features = extract_features(preprocessed, self._config.eeg.sampling_rate)
            inference = self._model.predict(features)
            inference = self._model.smooth(inference, self._last_inference)
            timestamp = float(frame.timestamps[-1]) if len(frame.timestamps) else time.time()
            directive_context = self._personalizer.ingest(
                raw_load=inference.load,
                confidence=inference.confidence,
                timestamp=timestamp,
            )
            directive = self._mapper.map(directive_context)
            if directive_context.suppress_adaptation:
                directive.metadata["note"] = "Directive suppressed due to low confidence or cadence guard."
            self._state = LoopState(directive=directive, inference=inference)
            self._last_inference = inference

    def _default_directive(self) -> AdaptationDirective:
        context = DirectiveContext(
            normalized_load=0.5,
            load_level="medium",
            confidence=0.0,
            trend="stable",
            suppress_adaptation=True,
            timestamp=time.time(),
        )
        return self._mapper.map(context)


async def run_demo_conversation(session: NeuroadaptiveSession) -> None:
    await session.start()
    try:
        print("Type messages to interact with the neuroadaptive assistant. Ctrl+C to exit.")
        while True:
            user_input = input("you> ")
            reply = await session.handle_user_message(user_input)
            print(f"ai> {reply}")
    except (EOFError, KeyboardInterrupt):
        print("\nStopping session...")
    finally:
        await session.shutdown()
