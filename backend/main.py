import os
import uuid
import tempfile
import time
import torch
import librosa
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.inference.pipeline import ViolenceDetectionPipeline
from src.config import SAMPLE_RATE
from src.utils.logger import get_logger

logger = get_logger("fastapi_backend")

# --- Pydantic Schemas ---
class StatusResponse(BaseModel):
    status: str

class UploadPingResponse(BaseModel):
    status: str
    filename: str

class SegmentResult(BaseModel):
    segment_index: int
    timestamp: str
    vad_speech: bool
    audio_score: float
    nlp_score: float
    transcript: str
    scream_text: bool
    scream_acoustic: bool
    temporal_trend: str
    temporal_segment_score: float
    state: str

class ProcessUploadResponse(BaseModel):
    file_path: str
    segments: List[SegmentResult]
    temporal_score: float
    final_state: str
# ------------------------

app = FastAPI(title="Hybrid Violence Detection API")

# Allow CORS for local React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Initializing Inference Pipeline...", extra={"extra_info": {"model": "tiny"}})

pipeline = ViolenceDetectionPipeline(whisper_model="tiny")
pipeline.load_weights()

@app.get("/", response_model=StatusResponse)
def read_root():
    return StatusResponse(status="Violence Detection Backend is running.")

@app.post("/ping_upload", response_model=UploadPingResponse)
async def ping_upload(audio: UploadFile = File(...)):
    return UploadPingResponse(status="upload_file_parser_works", filename=audio.filename or "unknown")

@app.post("/upload", response_model=ProcessUploadResponse)
async def process_upload(audio: UploadFile = File(...)):
    """Process a static audio file upload."""
    temp_dir = tempfile.gettempdir()
    # Handle the uploaded file extension
    ext = os.path.splitext(audio.filename)[1]
    if not ext:
        ext = ".wav"
    temp_file_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}{ext}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
    except Exception:
        import traceback
        logger.error(f"Upload stream read error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to read upload stream.")
        
    try:
        # Pass to the multi-modal inference pipeline
        results = pipeline.process_file(temp_file_path)
        
        # Pydantic will automatically validate and serialize this matching ProcessUploadResponse
        return ProcessUploadResponse(**results)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Pipeline failure: {str(e)}", extra={"extra_info": {"traceback": error_trace}})
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for live microphone streaming.
    Expects binary audio chunks (e.g. 4-second .webm or .wav chunks).
    """
    await websocket.accept()
    client_id = uuid.uuid4().hex
    logger.info("WebSocket client connected.", extra={"extra_info": {"client_id": client_id}})
    
    from src.models.temporal_tracker import RuleBasedTemporalTracker
    
    # Store session-specific temporal tracker
    session_tracker = RuleBasedTemporalTracker(window_size=5)
    
    try:
        while True:
            # Receive binary chunk from the frontend via websocket
            data = await websocket.receive_bytes()
            if not data:
                continue
                
            # Temporary cache to pass to librosa
            temp_file_path = os.path.join(tempfile.gettempdir(), f"stream_{uuid.uuid4().hex}.webm")
            with open(temp_file_path, "wb") as f:
                f.write(data)
                
            try:
                # Decode audio format
                y, sr = librosa.load(temp_file_path, sr=SAMPLE_RATE)
                
                from src.data.audio_utils import segment_audio, extract_mel_spectrogram
                from src.config import SEGMENT_LENGTH, N_MELS, N_FFT, HOP_LENGTH
                
                segments = segment_audio(y, sr, seg_len=SEGMENT_LENGTH)
                
                response = {"status": "processing"}
                
                for seg in segments:
                    # Execute multi-modal graph purely in RAM
                    mel_spec = extract_mel_spectrogram(seg, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(pipeline.device)
                    
                    with torch.no_grad():
                        audio_emb = pipeline.audio_encoder(mel_tensor)
                        
                    # Step 2b: FastVAD Check
                    has_speech, vad_details = pipeline.vad.has_speech(seg, sr)
                    
                    transcript = ""
                    threat_score = 0.0
                    
                    # Step 2c: Text branch
                    if has_speech:
                        transcript = pipeline._transcribe_segment(seg)
                        
                        # Filter out hallucinated punctuation & micro words like "." or "you"
                        import string
                        cleaned = transcript.translate(str.maketrans('', '', string.punctuation)).strip()
                        
                        if transcript and len(cleaned) >= 4:
                            text_emb = pipeline.text_encoder.get_embeddings([transcript]).to(pipeline.device)
                            threat_score = pipeline.text_encoder.get_threat_score(transcript)
                        else:
                            has_speech = False
                            transcript = transcript if transcript else ""
                            
                    if not has_speech:
                        text_emb = torch.zeros((1, 768), device=pipeline.device)
                        threat_score = 0.0
                    
                    # Step 2d: Scream check
                    scream_text = pipeline.scream_detector.detect(transcript) if transcript else False
                    scream_acoustic = pipeline.scream_detector.detect_acoustic(seg, sr)
                    is_impact = pipeline.scream_detector.detect_impact(seg, sr)
                    
                    # Step 2e: Adaptive Fusion
                    # We ALWAYS route to CMAG because CMAG holds the violence classification head.
                    with torch.no_grad():
                        seg_score = pipeline.cmag(audio_emb, text_emb, return_features=False)
                    baseline_score = seg_score.item()
                        
                    # Step 3: Temporal Tracking (Deterministic)
                    effective_score = baseline_score
                    if scream_acoustic or scream_text or threat_score > 0.6:
                        effective_score = max(effective_score, 0.85)
                    elif is_impact:
                        # Add a proportional boost for impact transients instead of flatlining
                        effective_score = min(effective_score + 0.25, 0.95)
                        
                    temporal_analysis = session_tracker.update(effective_score)
                    escalation_score = temporal_analysis["escalation_score"]
                    is_violent = escalation_score > 0.5
                    
                    response = {
                        "timestamp": time.strftime('%H:%M:%S'),
                        "audio_score": baseline_score,
                        "nlp_score": float(threat_score),
                        "transcript": transcript,
                        "scream_text": bool(scream_text),
                        "scream_acoustic": bool(scream_acoustic),
                        "temporal_score": escalation_score,
                        "temporal_trend": temporal_analysis["trend"],
                        "final_state": "VIOLENCE" if is_violent else "SAFE",
                        "status": "success"
                    }
                    
                    break # Usually streaming is exactly 1 segment per socket tick
                    
                await websocket.send_json(response)
                
            except Exception as inner_e:
                logger.error(f"Processing error: {str(inner_e)}", extra={"extra_info": {"client_id": client_id}})
                await websocket.send_json({"status": "error", "message": str(inner_e)})
            finally:
                if os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception:
                        pass
                    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.", extra={"extra_info": {"client_id": client_id}})
    except Exception as e:
        logger.error(f"Connection Error: {str(e)}", extra={"extra_info": {"client_id": client_id}})
