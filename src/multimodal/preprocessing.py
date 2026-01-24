"""多模态数据预处理"""
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    Resize,
    NormalizeIntensity,
)

from src.utils.logger import logger


class MultimodalPreprocessor:
    """
    多模态数据预处理器
    
    提供统一的数据预处理接口，包括：
    - 医学影像预处理
    - 文本预处理
    - 数值数据标准化
    - 时间序列预处理
    """
    
    def __init__(
        self,
        image_size: Optional[Union[int, Tuple[int, ...]]] = None,
        normalize_image: bool = True,
    ):
        """
        初始化预处理器
        
        Args:
            image_size: 图像目标尺寸
            normalize_image: 是否标准化图像
        """
        self.image_size = image_size
        self.normalize_image = normalize_image
        self._build_image_transforms()
    
    def _build_image_transforms(self) -> None:
        """构建图像变换管道"""
        transforms = [
            LoadImage(image_only=True),
            # MONAI 1.4+ 中 AddChannel 已弃用/移除，使用 EnsureChannelFirst 等价替代
            EnsureChannelFirst(channel_dim="no_channel"),
        ]
        
        if self.image_size:
            transforms.append(Resize(spatial_size=self.image_size))
        
        if self.normalize_image:
            transforms.extend([
                ScaleIntensity(),
                NormalizeIntensity(),
            ])
        
        self.image_transforms = Compose(transforms)
        logger.info("图像预处理管道已构建")
    
    def preprocess_image(
        self, image_path: Union[str, Path, np.ndarray]
    ) -> torch.Tensor:
        """
        预处理医学影像
        
        Args:
            image_path: 图像路径或数组
        
        Returns:
            预处理后的图像张量
        """
        if isinstance(image_path, (str, Path)):
            image = self.image_transforms(str(image_path))
        else:
            image = self.image_transforms(image_path)
        
        return image
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        预处理文本数据
        
        Args:
            text: 原始文本
        
        Returns:
            预处理后的文本字典（包含 token_ids, attention_mask 等）
        """
        # 基础文本清洗
        text = text.strip()
        text = " ".join(text.split())  # 规范化空白字符
        
        return {
            "text": text,
            "length": len(text),
        }
    
    def preprocess_lab_data(
        self, lab_data: Union[List, np.ndarray], method: str = "standard"
    ) -> torch.Tensor:
        """
        预处理实验室检查数据
        
        Args:
            lab_data: 实验室数据
            method: 标准化方法（'standard', 'minmax', 'robust'）
        
        Returns:
            预处理后的数据张量
        """
        if isinstance(lab_data, list):
            lab_data = np.array(lab_data)
        
        lab_data = lab_data.astype(np.float32)
        
        # 标准化
        if method == "standard":
            mean = np.mean(lab_data)
            std = np.std(lab_data)
            if std > 0:
                lab_data = (lab_data - mean) / std
        elif method == "minmax":
            min_val = np.min(lab_data)
            max_val = np.max(lab_data)
            if max_val > min_val:
                lab_data = (lab_data - min_val) / (max_val - min_val)
        
        return torch.tensor(lab_data, dtype=torch.float32)
    
    def preprocess_timeseries(
        self,
        timeseries: Union[List, np.ndarray],
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> torch.Tensor:
        """
        预处理时间序列数据
        
        Args:
            timeseries: 时间序列数据
            window_size: 滑动窗口大小
            stride: 滑动步长
        
        Returns:
            预处理后的时间序列张量
        """
        if isinstance(timeseries, list):
            timeseries = np.array(timeseries)
        
        timeseries = timeseries.astype(np.float32)
        
        # 如果指定了窗口大小，进行滑动窗口处理
        if window_size:
            if stride is None:
                stride = window_size
            
            windows = []
            for i in range(0, len(timeseries) - window_size + 1, stride):
                windows.append(timeseries[i : i + window_size])
            
            if windows:
                timeseries = np.array(windows)
            else:
                # 如果数据太短，进行填充
                padding = window_size - len(timeseries)
                timeseries = np.pad(
                    timeseries, (0, padding), mode="constant", constant_values=0
                )
                timeseries = timeseries.reshape(1, -1)
        
        return torch.tensor(timeseries, dtype=torch.float32)
    
    def preprocess_multimodal(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        预处理多模态数据
        
        Args:
            data: 包含多种模态数据的字典
        
        Returns:
            预处理后的多模态数据字典
        """
        processed = {}
        
        # 处理图像
        if "image" in data:
            processed["image"] = self.preprocess_image(data["image"])
        
        # 处理文本
        if "text" in data:
            processed["text"] = self.preprocess_text(data["text"])
        
        # 处理实验室数据
        if "lab_data" in data:
            processed["lab_data"] = self.preprocess_lab_data(data["lab_data"])
        
        # 处理时间序列
        if "timeseries" in data:
            processed["timeseries"] = self.preprocess_timeseries(
                data["timeseries"]
            )
        
        # 保留其他字段
        for key in ["label", "metadata"]:
            if key in data:
                processed[key] = data[key]
        
        return processed

