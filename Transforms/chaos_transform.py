# chaos_transform.py

from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
    CropForegroundd,
    RandShiftIntensityd,
    RandGaussianNoised,
    SpatialPadd,
    CenterSpatialCropd,
    AsDiscreted,
    NormalizeIntensityd
    EnsureTyped,
)

#
chaos_train_transforms = Compose([
    NormalizeIntensityd(keys=["image"]),

    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.01),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, 64)),
    AsDiscreted(keys=["label"], to_onehot=5),  # CHAOS has 5 channels: background + 4 organs
    EnsureTyped(keys=["image", "label"]),
])

chaos_val_transforms = Compose([
    ScaleIntensityRanged(
        keys=["image"],
        a_min=MRI_MIN,
        a_max=MRI_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, 64)),
    AsDiscreted(keys=["label"], to_onehot=5),
    EnsureTyped(keys=["image", "label"]),
])
