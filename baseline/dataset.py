import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SeismicDataset(Dataset):
    def __init__(self, data_dir, patch_size=(64, 64, 64), mode='train'):
        """
        data_dir: Path containing subfolders or files with valid X (cube) and Y (fault_mask).
                  Example structure:
                  data_dir/
                    ├── train/
                    │   ├── cube_1_X.npy
                    │   ├── cube_1_Y.npy
                    │   ├── cube_2_X.npy
        patch_size: Depth, Height, Width of the crop
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.mode = mode
        
        # Load all .npz file paths from the main folder
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        
    def __len__(self):
        # We define epoch len as a certain number of random crops from available volumes
        return len(self.files) * 5  # e.g., 5 random crops per volume per epoch

    def __getitem__(self, idx):
        # Pick a random volume
        vol_idx = idx % len(self.files)
        
        # Load full array into memory (128^3 is approx 16MB total, easily fits)
        data = np.load(self.files[vol_idx])
        x_vol = data['seis']
        y_vol = data['fault']

        # Random cropping params
        d, h, w = x_vol.shape
        pd, ph, pw = self.patch_size
        
        max_d = max(0, d - pd)
        max_h = max(0, h - ph)
        max_w = max(0, w - pw)
        
        start_d = np.random.randint(0, max_d + 1)
        start_h = np.random.randint(0, max_h + 1)
        start_w = np.random.randint(0, max_w + 1)
        
        # Crop
        x_patch = x_vol[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw].astype(np.float32)
        y_patch = y_vol[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw].astype(np.float32)
        
        # Normalize patch (Z-score normalisation locally or globally)
        # Here we do local standardization for the patch to counter amplitude variations
        mean = np.mean(x_patch)
        std = np.std(x_patch)
        if std > 1e-6:
            x_patch = (x_patch - mean) / std
        
        # Minimal augmentations (Gaussian noise)
        if self.mode == 'train' and np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.1, x_patch.shape).astype(np.float32)
            x_patch += noise

        # Add channel dimension
        x_patch = np.expand_dims(x_patch, axis=0)
        y_patch = np.expand_dims(y_patch, axis=0)

        return torch.tensor(x_patch), torch.tensor(y_patch)

def get_dataloader(data_dir, batch_size=4, patch_size=(64, 64, 64), mode='train', num_workers=2):
    dataset = SeismicDataset(data_dir, patch_size=patch_size, mode=mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
    return loader
