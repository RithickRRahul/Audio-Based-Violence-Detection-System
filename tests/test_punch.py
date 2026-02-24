import os
import sys
from pprint import pprint

# Ensure the src module is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference.pipeline import ViolenceDetectionPipeline

def test_punch_audio():
    file_path = "references/punch-sound-effects-28649.mp3"
    print(f"Testing new pipeline architecture on: {file_path}")
    print("-" * 50)
    
    # Initialize the pipeline
    pipeline = ViolenceDetectionPipeline()
    pipeline.load_weights()
    
    print("\nProcessing file...")
    
    # Let's temporarily inject a hook to see the audio_emb max
    audio_maxes = []
    original_audio_forward = pipeline.audio_encoder.forward
    def hooked_forward(x):
        emb = original_audio_forward(x)
        audio_maxes.append(float(import_torch().max(import_torch().abs(emb)).item()))
        return emb
        
    def import_torch():
        import torch
        return torch
        
    pipeline.audio_encoder.forward = hooked_forward
    
    results = pipeline.process_file(file_path)
    
    pipeline.audio_encoder.forward = original_audio_forward
    
    # Print the results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"File: {results['file_path']}")
    print(f"Final State: {results['final_state']}")
    print(f"Overall Temporal Score: {results['temporal_score']:.4f}")
    
    print("\nSegment Breakdown:")
    for idx, seg in enumerate(results['segments']):
        print(f"\n--- Segment {idx} ({idx*4}s to {(idx+1)*4}s) ---")
        print(f"Raw CMAG/Fusion Score: {seg['audio_score']:.4f}")
        print(f"Raw Audio Embedding Max: {audio_maxes[idx]:.4f}")
        print(f"Transcript (STT): '{seg['transcript']}'")
        print(f"Scream Detected: {seg['scream_acoustic'] or seg['scream_text']}")
        print(f"Rule-Based Temporal State: {seg['temporal_trend']}")
        print(f"Final Segment State: {seg['state']}")

if __name__ == '__main__':
    test_punch_audio()
