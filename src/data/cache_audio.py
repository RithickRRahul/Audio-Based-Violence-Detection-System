import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.datasets import build_combined_dataset

def cache_loader(loader, prefix):
    # Bypass the weighted sampler for caching to just iterate linearly
    dataset = loader.dataset
    
    # We use num_workers to utilize all CPU cores for audio processing
    cache_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=6)
    
    batch_idx = 0
    for mels, labels in tqdm(cache_loader, desc=f"Caching {prefix}"):
        torch.save((mels, labels), os.path.join(cache_dir, f"{prefix}_batch_{batch_idx}.pt"))
        batch_idx += 1

if __name__ == '__main__':
    print("Initializing datasets...")
    train_loader, test_loader, class_counts = build_combined_dataset(augment_train=True)

    cache_dir = "datasets/.cache_tensors"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Caching datasets to {cache_dir}...")
    cache_loader(train_loader, "train")
    cache_loader(test_loader, "test")

    print("Caching complete! Ensure train.py is updated to use these cached tensors if you want instant GPU loading.")
