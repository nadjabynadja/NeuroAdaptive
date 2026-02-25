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

# Width of the cognitive-state display panel (characters)
_PANEL_WIDTH = 62


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
        # Tracks the most recent *unsuppressed* directive so that
        # handle_user_message always injects an adaptive directive, even if the
        # most recent background frame happened to be suppressed by the cadence guard.
        self._last_applied_directive: Optional[AdaptationDirective] = None

    @property
    def state(self) -> LoopState:
        """Current loop state (directive + raw inference).  Read-only snapshot."""
        return self._state

    @property
    def applied_directive(self) -> Optional[AdaptationDirective]:
        """Last directive that was actually applied to an LLM call (not suppressed)."""
        return self._last_applied_directive

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
        # Prefer the last directive that was not suppressed by the cadence guard
        # or low-confidence check.  Fall back to the current state directive
        # (which may be suppressed), and finally to a safe default.
        directive = (
            self._last_applied_directive
            or self._state.directive
            or self._default_directive()
        )
        self._orchestrator.set_directive(directive)
        self._orchestrator.add_user_message(message)
        return await self._orchestrator.generate_response()

    async def _process_frames(self) -> None:
        async for frame in self._reader.frames():
            preprocessed = preprocess_frame(frame.eeg, self._config.eeg)
            features = extract_features(preprocessed, self._config.eeg.sampling_rate)
            inference = self._model.predict(features)
            inference = self._model.smooth(inference, self._last_inference)
            # Use real wall-clock time for the cadence guard so "2 seconds between
            # directive changes" means 2 real seconds, not 2 simulated EEG seconds.
            timestamp = time.time()
            directive_context = self._personalizer.ingest(
                raw_load=inference.load,
                confidence=inference.confidence,
                timestamp=timestamp,
            )
            directive = self._mapper.map(directive_context)
            if directive_context.suppress_adaptation:
                directive.metadata["note"] = "Directive suppressed due to low confidence or cadence guard."
            else:
                # Record this as the last directive that was actually applied.
                self._last_applied_directive = directive
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


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _panel_line(text: str, width: int = _PANEL_WIDTH) -> str:
    """Format a bordered panel line, padded to width."""
    inner = width - 4  # "│ " + content + " │"
    return f"│ {text:<{inner}} │"


def _wrap_instruction(instruction: str, inner_w: int) -> list[str]:
    words = instruction.split()
    line_buf: list[str] = []
    lines: list[str] = []
    for word in words:
        candidate = " ".join(line_buf + [word])
        if len(candidate) <= inner_w:
            line_buf.append(word)
        else:
            if line_buf:
                lines.append(" ".join(line_buf))
            line_buf = [word]
    if line_buf:
        lines.append(" ".join(line_buf))
    return lines


def print_state_panel(
    state: LoopState,
    *,
    applied_directive: Optional[AdaptationDirective] = None,
    baseline_ready: bool = True,
) -> None:
    """Print a framed cognitive-state summary to stdout.

    Args:
        state:            Current loop state (from session.state).
        applied_directive: Last directive injected into the LLM (from
                          session.applied_directive).  When supplied, it is
                          shown as the "LLM Directive" row; otherwise the
                          current state directive is used.
        baseline_ready:   Whether the baseline warmup has completed.
    """
    w = _PANEL_WIDTH
    top = "┌─ Cognitive State " + "─" * (w - 19) + "┐"
    sep = "├" + "─" * (w - 2) + "┤"
    bot = "└" + "─" * (w - 2) + "┘"

    print(top)

    if state.directive is None:
        print(_panel_line("(no directive yet — pipeline initialising)", w))
        print(bot)
        return

    meta = state.directive.metadata
    load_val = float(meta.get("load", 0.5))
    load_pct = int(load_val * 100)
    level = str(meta.get("load_level", "?")).upper()
    conf_pct = int(float(meta.get("confidence", 0.0)) * 100)
    trend = str(meta.get("trend", "?"))
    suppressed = meta.get("suppressed", False)

    raw_load = state.inference.load if state.inference else None
    raw_conf = state.inference.confidence if state.inference else None

    # Row 1: load level, trend, confidence
    load_str = f"Load: {level} ({load_pct}%)"
    trend_str = f"Trend: {trend}"
    conf_str = f"Confidence: {conf_pct}%"
    row1 = f"{load_str:<22}{trend_str:<20}{conf_str}"
    print(_panel_line(row1, w))

    # Row 2: raw inference values (useful for technical audiences)
    if raw_load is not None:
        raw_str = f"Raw inference:  load={raw_load:.3f}  conf={raw_conf:.3f}"
        print(_panel_line(raw_str, w))

    print(_panel_line("", w))

    # LLM Directive section — show what was actually injected into the model
    llm_dir = applied_directive or state.directive
    verb = llm_dir.verbosity_label
    tone = llm_dir.tone_label
    label = "LLM Directive" if applied_directive else "Directive"
    print(_panel_line(f"{label}:  verbosity={verb}  |  tone={tone}", w))

    # Instruction text (wrapped to panel width)
    inner_w = w - 4
    lines = _wrap_instruction(llm_dir.system_instruction, inner_w)
    for i, ln in enumerate(lines):
        prefix = 'Instruction: "' if i == 0 else "             "
        suffix = '"' if i == len(lines) - 1 else ""
        print(_panel_line(f"{prefix}{ln}{suffix}", w))

    if suppressed and applied_directive is not None:
        print(sep)
        print(_panel_line("Background frame suppressed (cadence guard); using last applied directive.", w))
    elif suppressed:
        note = str(meta.get("note", "Adaptation suppressed (low confidence or cadence guard)."))
        print(sep)
        print(_panel_line(f"NOTE: {note}", w))

    if not baseline_ready:
        print(sep)
        print(_panel_line("Calibration in progress — directives will personalise shortly.", w))

    print(bot)


# ---------------------------------------------------------------------------
# Interactive demo loop
# ---------------------------------------------------------------------------

async def run_demo_conversation(
    session: NeuroadaptiveSession,
    *,
    baseline_duration_seconds: float = 120.0,
) -> None:
    """Interactive neuroadaptive conversation loop.

    Uses run_in_executor so the EEG processing background task continues
    running (and updating session.state) while the user is typing.
    """
    await session.start()
    loop = asyncio.get_event_loop()

    try:
        print()
        print("╔" + "═" * (_PANEL_WIDTH - 2) + "╗")
        print("║" + " NeuroAdaptive Demo — Simulator Mode ".center(_PANEL_WIDTH - 2) + "║")
        print("║" + " Type a message and press Enter. Ctrl+C to exit. ".ljust(_PANEL_WIDTH - 2) + "║")
        print("╚" + "═" * (_PANEL_WIDTH - 2) + "╝")
        print()

        while True:
            # run_in_executor lets asyncio keep running EEG tasks while we wait
            user_input = await loop.run_in_executor(None, lambda: input("you> "))
            user_input = user_input.strip()
            if not user_input:
                continue

            reply = await session.handle_user_message(user_input)

            baseline_ready = session._personalizer.baseline_ready
            print()
            print_state_panel(
                session.state,
                applied_directive=session.applied_directive,
                baseline_ready=baseline_ready,
            )
            print()
            print(f"ai> {reply}")
            print()

    except (EOFError, KeyboardInterrupt):
        print("\nStopping session...")
    finally:
        await session.shutdown()


# ---------------------------------------------------------------------------
# Scripted demo (--demo flag)
# ---------------------------------------------------------------------------

_DEMO_SCRIPT = [
    {
        "phase": "RELAXED — Low Cognitive Load",
        "mode": "relaxed",
        "eeg_desc": (
            "Simulator: pure alpha (11–12 Hz). "
            "rel_alpha ≈ 0.95, engagement ≈ 0, theta_beta ≈ 0 → raw_load ≈ 0.01."
        ),
        "settle_seconds": 8,
        "message": "Can you walk me through how a neuroadaptive interface works?",
        "expected": "verbosity=high (low load → detailed response)",
    },
    {
        "phase": "OVERLOADED — High Cognitive Load",
        "mode": "overloaded",
        "eeg_desc": (
            "Simulator: pure high-beta/gamma (18–25 Hz). "
            "engagement >> 1, rel_alpha ≈ 0 → raw_load ≈ 1.0."
        ),
        "settle_seconds": 8,
        "message": (
            "Explain SSVEP, P300, and motor imagery BCIs: "
            "signal characteristics, decoding algorithms, practical limitations, "
            "and clinical applications."
        ),
        "expected": "verbosity=low, tone=high_load (high load → bullet points)",
    },
    {
        "phase": "FATIGUED — Medium / Falling Load",
        "mode": "fatigued",
        "eeg_desc": (
            "Simulator: theta (6–7 Hz) + alpha (11 Hz) + beta (15 Hz). "
            "Mixed bands → raw_load ≈ 0.50 (medium), trend falling from phase 2."
        ),
        "settle_seconds": 8,
        "message": "I need a break. What is the single most important thing to remember?",
        "expected": "verbosity=medium (medium load → balanced response)",
    },
]


async def run_scripted_demo(
    session: NeuroadaptiveSession,
    reader: EEGReader,
) -> None:
    """Run a pre-scripted 3-phase demo showing cognitive state transitions.

    Switches the EEG simulator to a new spectral profile before each phase,
    waits for the cognitive model to ingest frames from the new mode, sends
    a scripted user message, and displays the resulting directive and response.
    """
    # Pre-populate the baseline with a symmetric preset (mean=0.5, std=0.25) so
    # the personalisation layer is active from the first message.  This avoids
    # a 10-120 second warmup and makes the demo self-contained.
    session._personalizer.preset_demo_baseline(mean=0.5, std=0.25)

    await session.start()
    w = _PANEL_WIDTH

    print()
    print("╔" + "═" * (w - 2) + "╗")
    print("║" + " NeuroAdaptive — Scripted Demo ".center(w - 2) + "║")
    print("║" + " 3 phases: Relaxed  →  Overloaded  →  Fatigued ".center(w - 2) + "║")
    print("╚" + "═" * (w - 2) + "╝")
    print()

    directive_log: list[dict] = []

    try:
        for i, step in enumerate(_DEMO_SCRIPT, 1):
            phase_header = f"Phase {i}/{len(_DEMO_SCRIPT)}: {step['phase']}"
            print("┌" + "─" * (w - 2) + "┐")
            print("│" + f" {phase_header} ".center(w - 2) + "│")
            print("└" + "─" * (w - 2) + "┘")
            print(f"  EEG: {step['eeg_desc']}")
            print(f"  Expected: {step.get('expected', '')}")
            print()

            # Switch simulator spectral profile
            reader.set_mode(step["mode"])

            # Let the cognitive model ingest a few frames at the new profile
            settle = step["settle_seconds"]
            print(f"  Settling {settle}s for cognitive model to update …", end="", flush=True)
            await asyncio.sleep(settle)
            print(" ready.")
            print()

            # Send the scripted message
            print(f"  you> {step['message']}")
            reply = await session.handle_user_message(step["message"])

            baseline_ready = session._personalizer.baseline_ready
            print()
            print_state_panel(
                session.state,
                applied_directive=session.applied_directive,
                baseline_ready=baseline_ready,
            )
            print()
            print(f"  ai>  {reply}")
            print()

            # Record directive for end-of-demo summary — use the last applied
            # (unsuppressed) directive so the summary reflects what the LLM got.
            s = session.state
            applied = session.applied_directive or s.directive
            if applied:
                eeg_meta = s.directive.metadata if s.directive else {}
                directive_log.append(
                    {
                        "phase": step["phase"],
                        "load_level": eeg_meta.get("load_level", "?"),
                        "load_pct": int(float(eeg_meta.get("load", 0.5)) * 100),
                        "verbosity": applied.verbosity_label,
                        "tone": applied.tone_label,
                        "confidence": int(float(eeg_meta.get("confidence", 0.0)) * 100),
                    }
                )

            if i < len(_DEMO_SCRIPT):
                await asyncio.sleep(3)

    finally:
        await session.shutdown()

    # Summary table
    if directive_log:
        print()
        print("┌" + "─" * (w - 2) + "┐")
        print("│" + " Demo Summary — Directive Transitions ".center(w - 2) + "│")
        print("├" + "─" * (w - 2) + "┤")
        hdr = f"  {'Phase':<30}{'Load':<16}{'Verbosity':<12}{'Tone':<14}Conf"
        print(hdr)
        print("├" + "─" * (w - 2) + "┤")
        for d in directive_log:
            load_col = f"{d['load_level'].upper()} ({d['load_pct']}%)"
            row = f"  {d['phase'][:29]:<30}{load_col:<16}{d['verbosity']:<12}{d['tone']:<14}{d['confidence']}%"
            print(row)
        print("└" + "─" * (w - 2) + "┘")
        print()
