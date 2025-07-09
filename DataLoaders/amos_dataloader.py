import os
import numpy as np
from torch.utils.data import Dataset


class NPZVolumeDataset(Dataset):
    def __init__(self, folder, transforms=None):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])
        self.transforms = transforms
        self.organ_names = [
            "spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus",
            "liver", "stomach", "aorta", "inferior_vena_cava", "portal_splenic_vein",
            "pancreas", "right_adrenal_gland", "left_adrenal_gland", "duodenum", "bladder"
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.files[idx])
        data = np.load(file_path)

        image = data["data"].astype(np.float32)  # Shape: (H, W, D)
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape {image.shape}")

        label = np.zeros_like(image, dtype=np.uint8)
        for i, organ_name in enumerate(self.organ_names):
            mask_key = f"mask_{organ_name}"
            if mask_key in data:
                label[data[mask_key] == 1] = i + 1  # 0 is background

        sample = {
            "image": image[np.newaxis, ...],  # → shape (1, H, W, D)
            "label": label[np.newaxis, ...]   # → shape (1, H, W, D)
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample


# Optional test block
if __name__ == "__main__":
    dataset = NPZVolumeDataset(
        folder="/Users/dibya/PycharmProjects/Data/amos_npz"
    )
    sample = dataset[0]
    print("Test load:")
    print("Image shape:", sample["image"].shape)
    print("Label shape:", sample["label"].shape)
