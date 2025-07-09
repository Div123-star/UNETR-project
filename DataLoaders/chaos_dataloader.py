import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPZVolumeDataset(Dataset):
    def __init__(self, folder, transforms=None):
        self.folder = folder
        self.transforms = transforms
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.files[idx])
        data = np.load(file_path)

        image = data["data"].astype(np.float32)
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape {image.shape}")

        label = np.zeros_like(image, dtype=np.uint8)
        if "mask_Liver" in data:
            label[data["mask_Liver"] > 0] = 1
        if "mask_RK" in data:
            label[data["mask_RK"] > 0] = 2
        if "mask_LK" in data:
            label[data["mask_LK"] > 0] = 3
        if "mask_Spleen" in data:
            label[data["mask_Spleen"] > 0] = 4

        # ✅ Add channel dimension: shape becomes (1, H, W, D)
        image = image[np.newaxis, ...]
        label = label[np.newaxis, ...]

        sample = {"image": image, "label": label}

        if self.transforms:
            sample = self.transforms(sample)

        print("Image shape:", sample["image"].shape)
        print("Label shape:", sample["label"].shape)

        return sample





