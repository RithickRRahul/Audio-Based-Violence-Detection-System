# CMAG: Custom Multimodal Violence Detection System

A real-time violence detection pipeline designed for public safety and emergency response. It integrates environmental audio analysis with natural language processing to identify violent intent, physical impacts, and escalation trends with minimal latency.

The system utilizes a Custom Cross-Modal Attention Gate (CMAG) to adaptively weigh acoustic and textual modalities to determine threat levels across sliding temporal windows.

---

## URL / Source for Dataset

The system is trained and evaluated using a consolidated dataset derived from the following sources:

1.  **VSD (Violent Scene Detection)**: Used for identifying general aggressive and violent audio clips.
    *   Source: [MediaEval VSD Dataset](https://multimediaeval.github.io/2014-Violent-Scenes-Detection-Task/)
2.  **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**: Used for emotional speech recognition, mapping high-arousal "Angry" speech to violent intent.
    *   Source: [CREMA-D GitHub Repository](https://github.com/CheyneyComputerScience/CREMA-D)
3.  **ESC-50 (Dataset for Environmental Sound Classification)**: Provides 2,000 environmental recordings categorized into 50 classes, including violent categories like gunshots and breaking glass.
    *   Source: [ESC-50 GitHub Repository](https://github.com/karolpiczak/ESC-50)
4.  **UrbanSound8K**: Contains 8,732 labeled sound excerpts of urban sounds from 10 classes, used for classifying urban-based violent sounds.
    *   Source: [UrbanSound8K Dataset Page](https://urbansounddataset.weebly.com/urbansound8k.html)
5.  **Jigsaw Toxic Comment Classification Challenge**: Large-scale dataset of Wikipedia comments labeled for toxicity, used to train and benchmark the NLP/Text encoder for semantic violence detection.
    *   Source: [Kaggle Jigsaw Dataset Page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

---

## Hardware and Software Requirements

### Hardware Requirements
*   **Processor**: 4-core CPU (Intel i5 or AMD Ryzen 5 minimum).
*   **Memory**: 16 GB RAM recommended.
*   **GPU**: NVIDIA GPU with at least 8 GB VRAM (RTX 3060 or higher recommended) for real-time Speech-to-Text (STT) and Deep Modal Fusion.
*   **Storage**: 50 GB of available space for datasets and model weights.

### Software Requirements
*   **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+ recommended).
*   **Python**: Version 3.11 or higher.
*   **Node.js**: Version 18.x or higher (for the frontend application).
*   **System Tools**:
    *   **FFmpeg**: Must be installed and available in the system PATH for audio processing.
    *   **Tesseract OCR**: Required for secondary image-based analysis triggers.

---

## Detailed Instructions to Execute the Source Code

### 1. Repository Setup
Clone the repository and navigate into the project directory:
```bash
git clone <repository-url>
cd FYP
```

### 2. Backend Installation and Setup
Create a virtual environment and install the required Python dependencies:
```bash
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launching the Backend Server
Start the FastAPI server using Uvicorn:
```bash
python -m uvicorn backend.main:app --reload
```
The backend API will be accessible at `http://localhost:8000`.

### 4. Frontend Installation and Setup
Navigate to the frontend directory and install the necessary Node.js packages:
```bash
cd frontend
npm install
```

### 5. Launching the Frontend Application
Start the React development server:
```bash
npm run dev
```
The user interface will be available at `http://localhost:5173`.

### 6. Training the Fusion Model
To train the CMAG fusion layer using the consolidated datasets:
```bash
python scripts/run_transfer_learning.py
```

### 7. Evaluating System Performance
To run the full evaluation pipeline and generate performance metrics:
```bash
python scripts/evaluate_full_pipeline.py
```
This script will process the test split and output accuracy, precision, recall, and F1-score metrics for the system.

---

## Directory Structure

*   **/backend**: FastAPI server implementation.
*   **/frontend**: React (Vite) frontend application.
*   **/src**: Core AI implementation logic.
    *   **/data**: Audio processing and VAD utilities.
    *   **/models**: Neural network architectures (ResNet, RoBERTa, CMAG).
    *   **/inference**: End-to-end detection pipeline.
*   **/scripts**: Evaluation and training orchestration scripts.
*   **/tests**: Unit and integration tests for system verification.
