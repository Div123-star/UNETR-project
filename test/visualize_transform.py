import os
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    SpatialPadd,
    CenterSpatialCropd,
    EnsureTyped
)

# ----------- Paths -----------
npz_folder = "/Users/dibya/PycharmProjects/Data/amos_npz"
output_dir = "/Users/dibya/PycharmProjects/UNETR/test/TransformPreview"
os.makedirs(output_dir, exist_ok=True)

# ----------- Safe Preview Transform (No one-hot) -----------
preview_transform = Compose([
    NormalizeIntensityd(keys=["image"]),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
    RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.01),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, 64)),
    EnsureTyped(keys=["image", "label"]),
])

organ_names = [
    "spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus",
    "liver", "stomach", "aorta", "inferior_vena_cava", "portal_splenic_vein",
    "pancreas", "right_adrenal_gland", "left_adrenal_gland", "duodenum", "bladder"
]

# ----------- Loop Over Files -----------
for filename in sorted(os.listdir(npz_folder)):
    if not filename.endswith(".npz"):
        continue

    npz_path = os.path.join(npz_folder, filename)
    data = np.load(npz_path)

    image = data["data"].astype(np.float32)
    label = np.zeros_like(image, dtype=np.uint8)

    for i, organ_name in enumerate(organ_names):
        mask_key = f"mask_{organ_name}"
        if mask_key in data:
            label[data[mask_key] == 1] = i + 1

    sample = {
        "image": image[np.newaxis, ...],
        "label": label[np.newaxis, ...]
    }

    transformed = preview_transform(sample)
    trans_img = transformed["image"][0].detach().cpu().numpy()

    # Middle slice
    z = trans_img.shape[-1] // 2

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, z], cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(trans_img[:, :, z], cmap="gray")
    plt.title("Transformed")
    plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename.replace(".npz", "_preview.png"))
    plt.savefig(out_path)
    plt.close()

    print(f"Saved preview: {out_path}")

print("\n✅ All transform previews saved.")
