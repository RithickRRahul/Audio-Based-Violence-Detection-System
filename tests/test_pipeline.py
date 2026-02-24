import sys
import json
from src.inference.pipeline import ViolenceDetectionPipeline

try:
    print('Loading pipeline...')
    pipeline = ViolenceDetectionPipeline()
    pipeline.load_weights()
    print('Processing file...')
    res = pipeline.process_file('datasets/audio_violence/audios_VSD/audios_VSD/angry_01.wav')
    
    print('\nEvaluation Results:')
    print(f"Final file state: {res['final_state']}")
    print(f"File Temporal Score: {res['temporal_score']:.4f}")
    print('\nPer-Segment Analysis:')
    for seg in res['segments']:
        print(f"[{seg['timestamp']}] - State: {seg['state']} | CMAG: {seg['audio_score']:.4f} | LSTM: {seg['temporal_segment_score']:.4f} | NLP: {seg['nlp_score']:.4f} | Scream: {seg['scream_acoustic']}/{seg['scream_text']}")
        
except Exception as e:
    import traceback
    traceback.print_exc()
