# NeuroAdaptive — Demo Guide

## What NeuroAdaptive Does

NeuroAdaptive is a real-time brain-AI loop: it continuously reads EEG signals
(or simulates them), infers the user's cognitive load from spectral features,
translates that load into an *adaptation directive*, and injects the directive
into every LLM response — adjusting verbosity, tone, and structure without
any explicit user input.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No hardware required — the system runs entirely on the built-in EEG simulator.

**Optional — live LLM responses:**
```bash
export OPENAI_API_KEY="sk-..."   # without this the stub response runs offline
```

---

## Quick Start

```bash
# Interactive mode (10s baseline, then type freely)
python scripts/run_loop.py

# Scripted 3-phase demo (no typing needed — best for live presentations)
python scripts/run_loop.py --demo

# Longer baseline before directive personalisation kicks in
python scripts/run_loop.py --baseline-duration 30

# Reproducible run with fixed random seed
python scripts/run_loop.py --seed 42
```

---

## What You'll See

Every AI response is followed by a **Cognitive State panel**:

```
┌─ Cognitive State ─────────────────────────────────────────┐
│ Load: HIGH (78%)     Trend: rising     Confidence: 64%    │
│ Raw inference:  load=0.821  conf=0.640                    │
│                                                           │
│ Directive:  verbosity=low  |  tone=high_load              │
│ Instruction: "Provide concise bullet summaries. Use       │
│              reassuring, supportive language and offer    │
│              breaks."                                     │
├───────────────────────────────────────────────────────────┤
│ NOTE: Directive suppressed due to low confidence or       │
│       cadence guard.                                      │
└───────────────────────────────────────────────────────────┘
```

| Field | Meaning |
|-------|---------|
| **Load: LEVEL (%)** | Normalised cognitive load relative to your personal baseline. HIGH = more demanding than your rest state. |
| **Trend** | Direction of change over the last frame (rising / stable / falling). |
| **Confidence** | How much the system trusts the EEG inference. Below ~35% the directive is suppressed. |
| **Raw inference** | Pre-smoothing values from the cognitive model (useful for debugging). |
| **Directive: verbosity** | `low` = brief bullets (high load), `medium` = balanced, `high` = step-by-step (low load). |
| **Directive: tone** | `high_load` = reassuring/supportive, `engaged` = collaborative, `default` = neutral. |
| **Instruction** | Exact text prepended to the LLM system prompt for this turn. |
| **NOTE (suppressed)** | Cadence guard fired (< 2s since last directive) or confidence too low. |

---

## Scripted Demo Walkthrough (`--demo`)

The scripted demo runs three phases automatically, switching the EEG simulator
to a new spectral profile before each phase and waiting ~8 seconds for the
cognitive model to settle:

### Phase 1 — Relaxed (low load)
- **EEG profile:** Alpha-dominant, 8–11 Hz oscillations across all channels.
- **Key features:** High `rel_alpha`, low `engagement_index`, low `theta_beta_ratio`.
- **Expected directive:** `verbosity=high`, `tone=engaged` — the system assumes
  you have cognitive capacity and delivers a thorough, step-by-step response.
- **Scripted message:** *"Can you walk me through how a neuroadaptive interface works?"*

### Phase 2 — Overloaded (high load)
- **EEG profile:** Theta spike (5–6 Hz) + gamma activity (25–28 Hz).
- **Key features:** High `theta_beta_ratio`, high `engagement_index` (stress-driven).
- **Expected directive:** `verbosity=low`, `tone=high_load` — the system compresses
  to bullets and shifts to reassuring language.
- **Scripted message:** *"Explain SSVEP, P300, and motor imagery BCIs: signal
  characteristics, decoding algorithms, practical limitations, and clinical applications."*

### Phase 3 — Fatigued (falling load)
- **EEG profile:** Theta-dominant (6–8 Hz), weak beta.
- **Key features:** Rising `theta_beta_ratio`, falling `engagement_index`.
- **Expected directive:** `verbosity=medium`, `tone=high_load` — supportive, brief.
- **Scripted message:** *"I need a break. What is the single most important thing to remember?"*

At the end, a **summary table** compares the three directives side by side,
making the adaptation visible in one glance.

---

## Live Interaction — Suggested Topics

These prompts are calibrated to stress different cognitive states in the simulator.
With real EEG hardware they will reflect your actual load.

### Low Load (exploratory, receptive)
- "What are the main frequency bands in EEG and what do they represent?"
- "Explain the concept of oscillatory synchrony in neural circuits."
- "Walk me through the full signal-processing pipeline in this system."

### Medium Load (focused, engaged)
- "How does the personalisation layer differ from a simple threshold rule?"
- "What trade-offs does the heuristic cognitive model make compared to a trained classifier?"
- "How would you extend this system to adapt to emotional state, not just cognitive load?"

### High Load / Overload (complex multi-part questions)
- "Compare EEG, fNIRS, and fMRI for real-time BCI applications across dimensions of temporal resolution, spatial resolution, portability, and signal quality."
- "Describe the end-to-end architecture of a closed-loop neurofeedback system, including acquisition, preprocessing, decoding, feedback, and safety considerations."

### Fatigue / Wind-Down
- "What is the one-sentence summary of what just happened?"
- "I'm losing focus. Give me only the key takeaway."

---

## Pipeline Walkthrough

This is the 30-second version for a live presentation:

1. **EEG Acquisition** — Simulates (or reads) 4-channel EEG at 256 Hz, emitting
   2-second sliding windows every 0.5 seconds via an async queue.

2. **Preprocessing** — Band-pass filter (0.5–45 Hz) removes slow drift and EMG noise.
   Notch filter (50 Hz) removes power-line interference.  Z-score artifact suppression
   replaces gross outliers with the channel median.

3. **Feature Extraction** — FFT over the 2-second window extracts relative power in
   delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma
   (30–45 Hz) bands.  Derived features: theta-beta ratio, engagement index, spectral
   entropy, frontal asymmetry.

4. **Cognitive Inference** — A heuristic model (or pluggable sklearn artifact) maps
   features to a cognitive load score [0–1] and a confidence estimate.  Exponential
   smoothing (α = 0.6) prevents frame-to-frame jitter.

5. **Personalisation** — During the first N seconds a personal baseline is collected.
   After that, load is z-scored against the baseline and passed through a sigmoid,
   producing a normalised value independent of individual EEG amplitude differences.

6. **Directive Mapping** — Normalised load is categorised (low / medium / high).  The
   category, trend, and confidence determine which verbosity and tone instruction to
   issue.  A cadence guard prevents directives from changing faster than every 2 seconds.

7. **LLM Orchestration** — The directive instruction is prepended to the system prompt
   on every turn.  With an API key, the full message list is sent to the configured
   model.  Without one, a structured stub demonstrates the directive injection visibly.

---

## Cognitive State Reference

| EEG Profile | Dominant bands | theta_beta_ratio | engagement_index | Inferred Load | Directive |
|-------------|---------------|-----------------|-----------------|---------------|-----------|
| Relaxed | Alpha (8–13 Hz) | Low (< 1.0) | Low (< 0.8) | low | verbosity=high, tone=engaged |
| Focused | Beta (13–30 Hz) | Low–medium | High (> 1.2) | medium–high | verbosity=medium/low |
| Overloaded | Theta + Gamma | High (> 1.5) | High (stress) | high | verbosity=low, tone=high_load |
| Fatigued | Theta (4–8 Hz) | High (> 2.0) | Low (< 0.5) | medium (falling) | verbosity=medium, tone=high_load |

---

## Calibration

```bash
python scripts/calibration.py --duration 30
```

Collects 30 seconds of resting-state EEG, prints a human-readable interpretation,
and saves statistics to `baseline_stats.json`.

**30-second elevator pitch for non-neuroscientists:**
> "We're recording your brain's 'at rest' signature — a cognitive fingerprint.
> Later, when you're reading or problem-solving, we compare your live signal to
> this baseline.  'High cognitive load' means your brain is working harder than
> normal for *you*, not harder than some population average.  That personalisation
> is what makes the system adaptive rather than just rule-based."

---

## Talking Points by Pipeline Stage

**On EEG acquisition:**
> "We use BrainFlow, which abstracts over 30+ consumer and research-grade EEG
> devices. The same code runs on a $200 Muse headband or an $80k research amplifier.
> Today we're using the built-in physics simulator."

**On spectral features:**
> "The theta-beta ratio is one of the most replicated cognitive load biomarkers in
> the neuroscience literature.  High theta + low beta correlates with working memory
> saturation.  The engagement index (beta+gamma / alpha+theta) tracks the
> arousal-relaxation axis."

**On personalisation:**
> "Most BCI systems use population-level thresholds.  We track an online baseline
> for each user and z-score against it.  This accounts for baseline differences
> between people — someone who naturally has high alpha won't trigger 'low load'
> spuriously."

**On directive injection:**
> "The directive is just a string prepended to the LLM system prompt.  The model
> never knows the source is EEG — it just receives an instruction to be concise,
> or thorough, or reassuring.  The adaptation is entirely at the prompt level,
> which means it's model-agnostic."

**On the feedback loop:**
> "This is a one-way loop today: EEG → directive → response.  The natural extension
> is to close the loop: if a high-load directive reduces the user's load (measurable
> in the next few EEG frames), reinforce that directive.  If not, try a different
> intervention.  That's where it becomes a true adaptive system."

---

## Known Limitations / Honest Caveats

- **Heuristic model only.** No trained cognitive load classifier is included.
  The heuristic (engagement index + alpha suppression + theta-beta ratio) is
  reasonable but not validated on labelled data. Replace `model_path` in
  `CognitiveModelConfig` to plug in a joblib artifact.

- **Simulator ≠ real EEG.** The simulator generates clean sinusoids. Real EEG
  has ocular artifacts, muscle noise, movement artifacts, and inter-session
  variability. The preprocessing pipeline handles typical artifacts but has not
  been tested on real data at scale.

- **Single-user, single-session baseline.** The online baseline drifts slowly
  (adaptation_rate = 0.02) but does not persist across sessions. A production
  system would load a stored baseline on startup.

- **2-second window means 2-second latency.** The system infers cognitive state
  from the most recent 2-second window, updated every 0.5 seconds. This lag is
  acceptable for conversational pacing but not for real-time neurofeedback.

- **No ground truth.** We cannot validate directive quality in simulator mode.
  The value proposition requires real users, real tasks, and outcome measures
  (comprehension tests, task completion time, subjective load ratings).
