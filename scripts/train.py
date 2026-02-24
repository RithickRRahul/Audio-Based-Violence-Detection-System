"""
Master CLI wrapper for orchestrating the Hybrid Violence Detection System training phases.
"""
import argparse
import time
from src.config import LEARNING_RATE, EPOCHS
from src.training.train_audio import train_audio_encoder
from src.training.train_fusion import train_fusion
from src.training.train_temporal import train_temporal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid Violence Detection System")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["audio", "fusion", "temporal", "all"],
                        help="Training phase to run")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Max epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()
    
    start = time.time()
    
    if args.phase in ("audio", "all"):
        train_audio_encoder(epochs=args.epochs, lr=args.lr)
    
    if args.phase in ("fusion", "all"):
        train_fusion(epochs=min(args.epochs, 20), lr=args.lr)
    
    if args.phase in ("temporal", "all"):
        train_temporal(epochs=min(args.epochs, 20), lr=args.lr)
    
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Total training time: {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
