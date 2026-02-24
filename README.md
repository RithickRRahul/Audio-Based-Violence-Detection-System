# Hybrid Multimodal Violence Detection System

An enterprise-grade, real-time violence detection pipeline built for high-stakes societal impact (e.g., public safety, women's safety). It fuses environmental audio analysis with natural language processing to accurately identify violent intent, impacts, and escalation trends with near-zero latency.

## Core Capabilities
- **Real-Time Streaming:** Processes continuous microphone audio via WebSockets.
- **Multimodal Fusion:** Combines raw acoustic features (CMAG-v2) with semantic text understanding (Whisper + RoBERTa).
- **Transient Impact Detection:** Instantly identifies unvoiced violence like punches, glass breaking, or slaps using custom RMS and Delta heuristics.
- **Deterministic Temporal Tracking:** Evaluates risk across sliding temporal windows using strict "Spike," "Rising," and "Sustained" rules, rather than opaque sequence models.
- **Production-Ready Backend:** Hardened FastAPI infrastructure with strict Pydantic V2 validations and structured JSON observability logging.

## Architecture Overview

### 1. The Gateway: Fast-VAD
To prevent the system from wasting GPU cycles and hallucinating text on silent rooms or pure noise, all incoming audio is first routed through a **Rule-Based Voice Activity Detector (`FastVAD`)**.
- Uses Librosa's zero-crossing rate and spectral centroid.
- Bypasses computationally expensive Whisper STT if no speech is present, achieving zero natural language hallucination on pure background impact sounds.

### 2. The Twin Pillars: Audio & Text
If speech is detected:
- **Audio Encoder:** A custom ResNet-18 model processes 128-band Mel-Spectrograms to extract pure acoustic intent (tone, stress, environmental hazards).
- **Text Encoder:** OpenAI Whisper transcribes the speech, and a pretrained RoBERTa model extracts the semantic threat level (e.g., identifying verbal abuse or direct threats).

### 3. The Core: CMAG-v2 Adaptive Fusion
The Dual-Modal embeddings are fused via the **Cross-Modal Attention Gate (`CMAG-v2`)**. 
- Learns to weigh the acoustic modality versus the text modality adaptively.
- If `FastVAD` detects no speech, the NLP vector is dynamically zeroed out, forcing the network to evaluate the scenario purely on acoustic merit (impacts, shattering, scuffles).

### 4. The Critic: Temporal Tracker
Instead of relying on a black-box Bi-LSTM to interpret sequence history, the system uses a **Rule-Based Temporal Tracker**. 
- It maintains a 5-chunk (20-second) memory array and evaluates immediate danger using mathematical heuristics.
- **Spike Rule:** A massive jump in audio score immediately triggers a +25% alert penalty (e.g., a sudden gunshot).
- **Sustained Rule:** Continuous threats blend the current intensity with the historic peak, preventing the system from ignoring ongoing danger.

## Tech Stack
- **Deep Learning:** PyTorch, Torchaudio, Librosa
- **Web & API:** FastAPI, Pydantic V2, Uvicorn, WebSockets
- **Models:** Whisper (tiny), Custom ResNet-18, RoBERTa
- **Frontend:** React, Vite (Hot-Reloading UI)
- **Observability:** Custom Structured JSON Logger, Standardized Exception Bubbling
- **CI/CD & Quality:** Pytest, Ruff (PEP8), MyPy (Type Checking)

## Enterprise Quality & Safety Verification
Because of the sensitive nature of violence detection and public safety, this backend is rigorously engineered using several advanced verification methods:
1. **Targeted Unit Testing (TDD):** Every individual module (`fast_vad`, `temporal_tracker`, `scream_detector`) is verified against edge cases.
2. **Systematic Debugging:** Eliminates hardcoded temporal flatlining and prevents NLP score leakage on silence.
3. **Observability:** `logger.py` enforces structured JSON tracing for robust DataDog/ELK ingestion.
4. **Linting and Typing:** `ruff` and `mypy` statically enforce Type safety to prevent silent inference failures in production.
