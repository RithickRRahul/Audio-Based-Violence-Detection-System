import os
import torch
from torch.utils.data import Dataset, DataLoader

class CachedTensorDataset(Dataset):
    """
    A lightweight PyTorch Dataset that loads pre-computed .pt tensors
    (mel-spectrograms and labels) directly from disk.
    Drastically speeds up ResNet-18 training on large datasets.
    """
    def __init__(self, cache_dir: str, prefix: str):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.files = [f for f in os.listdir(cache_dir) if f.startswith(f"{prefix}_batch_") and f.endswith(".pt")]
        
        # We need to map global indices to specific batch files and offsets
        self.index_map = []
        
        print(f"[Cache Load] Indexing {prefix} dataset...")
        for file in sorted(self.files):
            file_path = os.path.join(cache_dir, file)
            # Load just to get the shape/length, we don't keep it in memory
            # to avoid using 10GB+ of system RAM.
            mels, labels = torch.load(file_path, weights_only=True)
            batch_size = mels.size(0)
            
            for i in range(batch_size):
                self.index_map.append((file_path, i))
                
        print(f"  -> Indexed {len(self.index_map)} {prefix} samples.")
        
        # Cache the current loaded batch to avoid hitting disk 64 times for the same file
        self.current_loaded_file = None
        self.current_mels = None
        self.current_labels = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, local_idx = self.index_map[idx]
        
        if self.current_loaded_file != file_path:
            self.current_mels, self.current_labels = torch.load(file_path, weights_only=True)
            self.current_loaded_file = file_path
            
        return self.current_mels[local_idx], self.current_labels[local_idx]

def get_cached_dataloaders(cache_dir: str = "datasets/.cache_tensors", batch_size: int = 32):
    train_ds = CachedTensorDataset(cache_dir, "train")
    test_ds = CachedTensorDataset(cache_dir, "test")
    
    # We don't need num_workers > 0 here because disk IO of .pt files is extremely fast
    # and using workers with the internal state caching architecture causes memory leaks.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
