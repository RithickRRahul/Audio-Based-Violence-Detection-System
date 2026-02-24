# CMAG-v2: Custom Multimodal Violence Detection System

An enterprise-grade, real-time violence detection pipeline built for high-stakes societal impact (e.g., public safety, women's safety). It fuses environmental audio analysis with natural language processing to accurately identify violent intent, physical impacts, and escalation trends with near-zero latency.

## 🚀 The Core Problem & Solution

Traditional audio surveillance relies solely on volume thresholds or basic keyword matching. This system solves complex, real-world scenarios by combining acoustic heuristics (the *sound* of violence, like a punch or glass breaking) with semantic intent (the *meaning* of spoken words, like threats or abuse). 

By utilizing a **Custom Cross-Modal Attention Gate (CMAG-v2)**, the system adaptively weighs the acoustic modality versus the text modality to determine the true threat level, evaluating danger deterministically across sliding temporal windows.

## 🧠 System Architecture

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

## ☁️ Deployment Architecture (Vercel & Inference)

### Handling Models & The Cloud
**Machine Learning models (`saved_models/`) are NOT uploaded to this GitHub repository.** 

Why? Because GitHub is designed for text/code, not large >50MB binary tensors. 

**So how do we deploy to Vercel?**
Vercel is strictly a **Frontend / Serverless** hosting provider. They do not have the GPU capacity or persistent disk storage required to run heavy PyTorch inference pipelines natively.

**The Production Solution:**
1.  **Backend (The AI Engine):** The FastAPI code (`/backend` + `/src` + models) is hosted on a dedicated cloud provider built for ML inference (e.g., Render, Railway, AWS EC2, or Google Cloud Run). The heavy models are stored securely in cloud storage (like AWS S3) and downloaded locally by the backend server when it boots up.
2.  **Frontend (The UI):** The React interface (`/frontend`) is hooked up to GitHub and deployed directly onto Vercel. 
3.  **The Link:** The deployed Vercel frontend is configured via Environment Variables to send its audio streams directly to the IP address of your dedicated Backend Inference Server!
