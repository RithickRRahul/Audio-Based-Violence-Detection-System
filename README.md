# CMAG-v2: Custom Multimodal Violence Detection System

A real-time violence detection pipeline built for high-stakes societal impact (e.g., public safety, women's safety). It fuses environmental audio analysis with natural language processing to accurately identify violent intent, physical impacts, and escalation trends with near-zero latency.

## 🚀 The Core Problem & Solution

Traditional audio surveillance relies solely on volume thresholds or basic keyword matching. This system solves complex, real-world scenarios by combining acoustic heuristics (the *sound* of violence, like a punch or glass breaking) with semantic intent (the *meaning* of spoken words, like threats or abuse). 

By utilizing a **Custom Cross-Modal Attention Gate (CMAG-v2)**, the system adaptively weighs the acoustic modality versus the text modality to determine the true threat level, evaluating danger deterministically across sliding temporal windows.

## 🧠 System Architecture

```mermaid
graph TD
    %% Define Styles
    classDef input fill:#2d3436,stroke:#b2bec3,stroke-width:2px,color:#fff
    classDef gateway fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    classDef branch fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    classDef core fill:#d63031,stroke:#fab1a0,stroke-width:2px,color:#fff
    classDef output fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
    classDef fallback fill:#e17055,stroke:#ffeaa7,stroke-width:2px,color:#fff

    %% Components
    A["🎤 Live Audio Stream (WebSockets / Files)"]:::input
    B{"FastVAD (Librosa)"}:::gateway
    
    %% Audio Branch
    C["Mel-Spectrogram Extraction"]:::branch
    D["Audio Encoder (ResNet-18)"]:::branch
    
    %% Text / NLP Branch
    E{"Speech Detected?"}:::gateway
    F["Whisper (STT)"]:::branch
    G["Text Encoder (RoBERTa)"]:::branch
    H["Scream/Impact Failsafe"]:::fallback
    
    %% Fusion and Output
    I{"CMAG-v2\n(Cross-Modal Attention Gate)"}:::core
    J["Deterministic Temporal Tracker\n(Spike, Rising, Sustained)"]:::core
    K["🛡️ Final Verdict\n[SAFE / VIOLENCE]"]:::output

    %% Flow
    A --> B
    B --> C
    C --> D
    
    B --> E
    E -- "Yes" --> F
    F --> G
    F -.-> H
    E -- "No" --> |Silent NLP Vector| I
    C -.-> H
    
    D --> I
    G --> I
    H -- "Override Signal" --> J
    I --> J
    J --> K
```

### 1. The Gateway: Fast-VAD
To prevent the system from wasting GPU cycles and hallucinating text on silent rooms or pure noise, incoming audio is first routed through a **Rule-Based Voice Activity Detector (`FastVAD`)**.
*   Bypasses computationally expensive STT if no speech is present.
*   Ensures structural integrity by preventing NLP models from "guessing" at impact sounds.

### 2. The Twin Pillars: Audio & Text
If speech is detected:
*   **Audio Encoder:** A custom ResNet-18 model processes 128-band Mel-Spectrograms to extract pure acoustic intent (tone, stress, environmental hazards).
*   **Text Encoder:** OpenAI Whisper transcribes the speech, and a pretrained RoBERTa model extracts semantic threat levels.

### 3. The Core: CMAG-v2 Adaptive Fusion
The Dual-Modal embeddings are fused via the **Cross-Modal Attention Gate (`CMAG-v2`)**.
*   Dynamically learns to weigh the acoustic modality against the text modality.
*   Features a fallback `ScreamDetector` to catch primal, unvoiced distress directly.

### 4. The Critic: Temporal Tracker
The system uses a **Deterministic Rule-Based Temporal Tracker** maintaining a 5-chunk (20-second) memory array.
*   **Spike Rule:** Immediate jumps in threat score trigger escalation.
*   **Sustained Rule:** Continuous lower-level threats blend to prevent the system from normalizing ongoing danger.

---

## 📂 Directory Structure Guide

Understanding the repository layout:

*   **`/backend`**: Contains the FastAPI server (`main.py`) which acts as the bridge between the AI pipeline and the web frontend.
*   **`/frontend`**: A React application (built with Vite) serving the interactive UI for Live Mic streaming and File Uploads.
*   **`/src`**: The core AI logic.
    *   `/data`: Audio utilities and VAD logic.
    *   `/models`: Neural network definitions (CMAG, ResNet-18, RoBERTa encoders).
    *   `/inference`: The end-to-end `pipeline.py` that ties all models together.
*   **`/tests`**: Comprehensive unit and integration tests (Pytest) to ensure mathematical and logical integrity.
*   **`/tools`**: Helper utility scripts, such as metric generators.
*   **`/docs`**: Evaluation metrics, performance reports, and generated graphs proving the system's efficacy against standard baselines (like YAMNet).
*   **`/scripts`**: Top-level execution and database scripts.

*(Note: `saved_models/`, `datasets/`, and `logs/` are explicitly set to be ignored by Git to keep the repository lightweight and fast. See "Handling Models" below).*

---

## 🛠️ Getting Started (Local Development)

### Prerequisites
*   Python 3.11+
*   Node.js (v18+)
*   FFmpeg (must be installed on your system path for audio processing)

### 1. Clone the Repository
```bash
git clone <YOUR-GITHUB-URL>
cd FYP
```

### 2. Backend Setup (AI Engine)
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# Activate the environment:
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

Start the FastAPI application:
```bash
python -m uvicorn backend.main:app --reload
```
*The backend will boot up at `http://localhost:8000`.*

### 3. Frontend Setup (Web UI)
Open a new terminal window, navigate to the frontend directory, and install Node modules:
```bash
cd frontend
npm install
```

Start the React development server:
```bash
npm run dev
```
*The web interface will be available at `http://localhost:5173`. You can now use the Live Mic or upload `.wav` files!*

---

## 🧪 Testing and Tooling

### Running the Test Suite
The repository includes a rigorous suite of 34 tests covering model architectures, pipeline integration, and mathematical precision limits. To verify system integrity:
```bash
pytest tests/ -v
```

### Generating Performance Proof
To generate the visualizations proving CMAG-v2's capabilities against baselines:
```bash
python scripts/simulate_yamnet_metrics.py
```
*Graphs will be generated inside `docs/performance_metrics_yamnet/`.*

---

