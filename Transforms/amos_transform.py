from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
    CropForegroundd,
    RandShiftIntensityd,
    RandGaussianNoised,
    SpatialPadd,
    CenterSpatialCropd,
    AsDiscreted,
    EnsureTyped,
    NormalizeIntensityd,
    RandCropByPosNegLabeld
)

amos_train_transforms = Compose([
    NormalizeIntensityd(keys=["image"]),
    # CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.01),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, 64)),
    AsDiscreted(keys=["label"], to_onehot=15),
    EnsureTyped(keys=["image", "label"]),
])

amos_val_transforms = Compose([
    NormalizeIntensityd(keys=["image"]),
    SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, 64)),
    AsDiscreted(keys=["label"], to_onehot=15),
    EnsureTyped(keys=["image", "label"]),
])
