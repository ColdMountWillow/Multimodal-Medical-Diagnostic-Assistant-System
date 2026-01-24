"""医学影像数据变换"""
from typing import Optional, Tuple
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    NormalizeIntensity,
    Resize,
    RandRotate,
    RandFlip,
    RandZoom,
    RandShiftIntensity,
    RandGaussianNoise,
    ToTensor,
)


def get_train_transforms(
    spatial_size: Optional[Tuple[int, ...]] = None,
    use_augmentation: bool = True,
) -> Compose:
    """
    获取训练时的数据变换
    
    Args:
        spatial_size: 目标空间尺寸
        use_augmentation: 是否使用数据增强
    
    Returns:
        MONAI Compose 变换
    """
    transforms = [
        LoadImage(image_only=True),
        # MONAI 1.4+ 中 AddChannel 已弃用/移除，使用 EnsureChannelFirst 等价替代
        EnsureChannelFirst(channel_dim="no_channel"),
    ]
    
    if spatial_size:
        transforms.append(Resize(spatial_size=spatial_size))
    
    transforms.extend([
        ScaleIntensity(),
        NormalizeIntensity(),
    ])
    
    if use_augmentation:
        transforms.extend([
            RandRotate(prob=0.5, range_x=0.1),
            RandFlip(prob=0.5, spatial_axis=0),
            RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.1),
            RandShiftIntensity(prob=0.5, offsets=0.1),
            RandGaussianNoise(prob=0.3, std=0.1),
        ])
    
    transforms.append(ToTensor())
    
    return Compose(transforms)


def get_val_transforms(
    spatial_size: Optional[Tuple[int, ...]] = None,
) -> Compose:
    """
    获取验证时的数据变换
    
    Args:
        spatial_size: 目标空间尺寸
    
    Returns:
        MONAI Compose 变换
    """
    transforms = [
        LoadImage(image_only=True),
        # MONAI 1.4+ 中 AddChannel 已弃用/移除，使用 EnsureChannelFirst 等价替代
        EnsureChannelFirst(channel_dim="no_channel"),
    ]
    
    if spatial_size:
        transforms.append(Resize(spatial_size=spatial_size))
    
    transforms.extend([
        ScaleIntensity(),
        NormalizeIntensity(),
        ToTensor(),
    ])
    
    return Compose(transforms)


def get_inference_transforms(
    spatial_size: Optional[Tuple[int, ...]] = None,
) -> Compose:
    """
    获取推理时的数据变换
    
    Args:
        spatial_size: 目标空间尺寸
    
    Returns:
        MONAI Compose 变换
    """
    return get_val_transforms(spatial_size)

