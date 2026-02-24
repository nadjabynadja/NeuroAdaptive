import argparse
import asyncio
import os
import sys

# Allow running as `python scripts/run_loop.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroadaptive.cognitive_state import CognitiveStateModel
from neuroadaptive.config import SystemConfig
from neuroadaptive.directives import DirectiveMapper
from neuroadaptive.eeg_acquisition import EEGReader
from neuroadaptive.llm_orchestrator import LLMOrchestrator
from neuroadaptive.neuro_loop import (
    NeuroadaptiveSession,
    run_demo_conversation,
    run_scripted_demo,
)
from neuroadaptive.personalization import PersonalizationManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroAdaptive real-time brain-AI loop demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_loop.py                        # interactive, baseline=10s
  python scripts/run_loop.py --demo                 # scripted 3-phase demo
  python scripts/run_loop.py --baseline-duration 5  # very short baseline
  python scripts/run_loop.py --baseline-duration 120 --seed 42  # production run
        """,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the scripted 3-phase demo (Relaxed → Overloaded → Fatigued) "
             "without requiring live user input.",
    )
    parser.add_argument(
        "--baseline-duration",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Seconds of EEG data to collect before the personalised baseline "
             "activates.  Default: 10 (demo-friendly). Production: 120.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for the EEG simulator (reproducible runs).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    config = SystemConfig()
    # Override baseline duration from CLI so demos don't wait 2 minutes
    config.personalization.baseline_duration_seconds = int(args.baseline_duration)

    reader = EEGReader(config.eeg, use_simulator=True, simulator_seed=args.seed)
    cognitive_model = CognitiveStateModel(config.cognitive_model)
    personalizer = PersonalizationManager(
        config.personalization, step_size_seconds=config.eeg.step_size_seconds
    )
    directive_mapper = DirectiveMapper(config.directives)
    orchestrator = LLMOrchestrator(config.llm)

    session = NeuroadaptiveSession(
        config=config,
        reader=reader,
        cognitive_model=cognitive_model,
        personalizer=personalizer,
        directive_mapper=directive_mapper,
        orchestrator=orchestrator,
    )

    if args.demo:
        await run_scripted_demo(session, reader)
    else:
        await run_demo_conversation(
            session,
            baseline_duration_seconds=args.baseline_duration,
        )


if __name__ == "__main__":
    asyncio.run(main())
