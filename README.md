# NeuralReader Neuroadaptive Loop

This prototype implements a real-time brain-AI loop that uses EEG signals to adapt large language model (LLM) responses to the user's cognitive state. It targets consumer EEG headsets (Muse, OpenBCI) via the BrainFlow SDK and includes a simulator so the stack runs without hardware.

## Components
- **EEG acquisition:** Async BrainFlow reader with a physics-inspired simulator for development.
- **Signal processing:** Band-pass and notch filtering, simple artifact suppression, and spectral feature extraction.
- **Cognitive inference:** Pluggable model (joblib artifact or heuristic fallback) with smoothing.
- **Personalization:** Online baseline tracking, load normalization, and guardrails for directives.
- **Directive mapping:** Translates cognitive context into LLM instructions (verbosity, tone, guard rails).
- **LLM orchestration:** Prompt wrapper for OpenAI-compatible chat models with a local stub for offline testing.
- **Scripts:** `run_loop.py` for interactive demos, `calibration.py` for baseline collection.

## Getting Started
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. (Optional) Set your OpenAI-compatible API key if you want live LLM generations:
   ```bash
   setx OPENAI_API_KEY "sk-..."   # Windows
   export OPENAI_API_KEY="sk-..."   # Linux/macOS
   ```
3. Run the neuroadaptive demo in simulator mode:
   ```bash
   python scripts/run_loop.py
   ```
   Type into the prompt and observe how the stubbed response includes the active directive metadata.

## Using Real Hardware
1. Install the BrainFlow drivers and note the board ID and connection parameters for your device.
2. Replace the `EEGReader` instantiation in `scripts/run_loop.py` with:
   ```python
   from brainflow.board_shim import BrainFlowInputParams

   params = BrainFlowInputParams()
   params.serial_port = "COM3"  # or your device path
   reader = EEGReader(config.eeg, params=params, use_simulator=False)
   ```
3. Run `python scripts/calibration.py --duration 180` to collect a personalized baseline while the user performs relaxed and focused tasks.
4. Save the resulting `baseline_stats.json` and load it in your personalization manager or extend `PersonalizationManager` to bootstrap from stored baselines.

## Extending the System
- Train a cognitive load classifier and export a `joblib` artifact with `model`, `scaler`, and `feature_keys` entries to plug into `CognitiveStateModel`.
- Integrate the directive metadata with downstream LLM prompt templates or tool selection logic.
- Add a dashboard (e.g., FastAPI + Plotly) to visualize load trends and manual overrides.
- Schedule re-calibration prompts and add supervised feedback to refine personalization over time.

## Safety Notes
- Always secure raw EEG data and derivative metrics; they are sensitive biometric signals.
- Keep guardrails enabled so the assistant does not whiplash between behaviors on noisy data.
- Provide clear user controls to pause adaptation and review recorded data.
