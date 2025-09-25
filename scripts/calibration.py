import argparse
import asyncio
import json
import time

from neuroadaptive.cognitive_state import CognitiveStateModel
from neuroadaptive.config import SystemConfig
from neuroadaptive.eeg_acquisition import EEGReader
from neuroadaptive.personalization import PersonalizationManager
from neuroadaptive.preprocessing import preprocess_frame
from neuroadaptive.features import extract_features


async def calibrate(duration: float, output_path: str) -> None:
    config = SystemConfig()
    reader = EEGReader(config.eeg, use_simulator=True)
    model = CognitiveStateModel(config.cognitive_model)
    personalizer = PersonalizationManager(config.personalization, step_size_seconds=config.eeg.step_size_seconds)
    await reader.start()
    start = time.time()
    last_inference = None
    try:
        async for frame in reader.frames():
            preprocessed = preprocess_frame(frame.eeg, config.eeg)
            features = extract_features(preprocessed, config.eeg.sampling_rate)
            inference = model.predict(features)
            inference = model.smooth(inference, last_inference)
            personalizer.ingest(inference.load, inference.confidence)
            last_inference = inference
            if time.time() - start >= duration:
                break
    finally:
        await reader.stop()
    summary = personalizer.baseline_summary()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Baseline saved to {output_path} -> {summary}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Collect baseline EEG calibration data using simulator or hardware.")
    parser.add_argument("--duration", type=float, default=120.0, help="Calibration duration in seconds.")
    parser.add_argument("--output", type=str, default="baseline_stats.json", help="Output JSON file path.")
    args = parser.parse_args()
    await calibrate(args.duration, args.output)


if __name__ == "__main__":
    asyncio.run(main())
