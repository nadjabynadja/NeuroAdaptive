import argparse
import asyncio
import json
import os
import sys
import time

# Allow running as `python scripts/calibration.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroadaptive.cognitive_state import CognitiveStateModel
from neuroadaptive.config import SystemConfig
from neuroadaptive.eeg_acquisition import EEGReader
from neuroadaptive.features import extract_features
from neuroadaptive.personalization import PersonalizationManager
from neuroadaptive.preprocessing import preprocess_frame


def _interpret_baseline(summary: dict, duration: float, step_size: float) -> str:
    """Return a plain-English interpretation of the baseline statistics.

    Intended for a 30-second verbal explanation to a non-neuroscientist.
    """
    mean = summary.get("mean", 0.5)
    std = summary.get("std", 0.0)
    count = int(summary.get("count", 0))
    seconds = count * step_size

    # Qualitative buckets
    if mean < 0.35:
        mean_label = "LOW  (below median — system is in a relaxed resting state)"
    elif mean < 0.65:
        mean_label = "MODERATE  (near median — typical resting baseline)"
    else:
        mean_label = "HIGH  (above median — elevated baseline activity)"

    if std < 0.05:
        std_label = "STABLE  (very consistent signal across the recording)"
    elif std < 0.15:
        std_label = "MODERATE  (normal session-to-session variability)"
    else:
        std_label = "VARIABLE  (high drift — consider a longer or cleaner recording)"

    lines = [
        "",
        f"  Baseline collected from {count} frames (~{seconds:.0f}s of EEG).",
        "",
        f"  Mean resting load index : {mean:.3f}  →  {mean_label}",
        f"  Signal variability (std): {std:.3f}  →  {std_label}",
        "",
        "  What this means (non-specialist):",
        "    The system just recorded your brain's 'at rest' cognitive signature —",
        "    a kind of personal baseline.  Future load estimates are normalised",
        "    against this fingerprint, so 'high load' means elevated relative to",
        "    YOUR resting state, not an absolute threshold.  This personalisation",
        "    is what separates a neuroadaptive system from a simple threshold rule.",
        "",
        "    Low variability means the signal was consistent, which is expected in",
        "    simulator mode.  With real EEG, variability reflects genuine fluctuations",
        "    in attention and arousal across the recording window.",
        "",
    ]
    return "\n".join(lines)


async def calibrate(duration: float, output_path: str) -> None:
    config = SystemConfig()
    step_size = config.eeg.step_size_seconds
    reader = EEGReader(config.eeg, use_simulator=True)
    model = CognitiveStateModel(config.cognitive_model)
    personalizer = PersonalizationManager(config.personalization, step_size_seconds=step_size)

    print(f"\nCalibrating for {duration:.0f}s (use Ctrl+C to stop early)…")
    print("  Collecting resting-state EEG frames …", end="", flush=True)

    await reader.start()
    start = time.time()
    last_inference = None
    frame_count = 0
    try:
        async for frame in reader.frames():
            preprocessed = preprocess_frame(frame.eeg, config.eeg)
            features = extract_features(preprocessed, config.eeg.sampling_rate)
            inference = model.predict(features)
            inference = model.smooth(inference, last_inference)
            personalizer.ingest(inference.load, inference.confidence)
            last_inference = inference
            frame_count += 1
            elapsed = time.time() - start
            if frame_count % 10 == 0:
                pct = min(100, int(elapsed / duration * 100))
                print(f"\r  Collecting resting-state EEG frames … {pct:3d}%", end="", flush=True)
            if elapsed >= duration:
                break
    finally:
        await reader.stop()

    print("\r  Collecting resting-state EEG frames … done.   ")

    summary = personalizer.baseline_summary()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(_interpret_baseline(summary, duration, step_size))
    print(f"  Saved → {output_path}")
    print()
    print("  Next step: run the demo loop.")
    print("    python scripts/run_loop.py")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a resting-state EEG baseline for the NeuroAdaptive system.\n\n"
            "For a demo, 30 seconds is sufficient.  For production use with real\n"
            "hardware, 120 seconds gives a more stable personal baseline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Calibration duration in seconds (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_stats.json",
        help="Output JSON file path (default: baseline_stats.json).",
    )
    args = parser.parse_args()
    await calibrate(args.duration, args.output)


if __name__ == "__main__":
    asyncio.run(main())
