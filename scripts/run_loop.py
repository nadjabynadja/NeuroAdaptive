import asyncio

from neuroadaptive.cognitive_state import CognitiveStateModel
from neuroadaptive.config import SystemConfig
from neuroadaptive.directives import DirectiveMapper
from neuroadaptive.eeg_acquisition import EEGReader
from neuroadaptive.llm_orchestrator import LLMOrchestrator
from neuroadaptive.neuro_loop import NeuroadaptiveSession, run_demo_conversation
from neuroadaptive.personalization import PersonalizationManager


async def main() -> None:
    config = SystemConfig()
    reader = EEGReader(config.eeg, use_simulator=True)
    cognitive_model = CognitiveStateModel(config.cognitive_model)
    personalizer = PersonalizationManager(config.personalization, step_size_seconds=config.eeg.step_size_seconds)
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
    await run_demo_conversation(session)


if __name__ == "__main__":
    asyncio.run(main())
