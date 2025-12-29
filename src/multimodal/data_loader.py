"""多模态数据加载器"""
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
from monai.data import DataLoader as MonaiDataLoader
from monai.transforms import Compose

from src.utils.logger import logger


class MultimodalDataset(Dataset):
    """
    多模态数据集类
    
    支持加载多种医疗数据类型：
    - 医学影像（CT、MRI、X光片、病理切片）
    - 病历文本
    - 实验室检查数据
    - 生理信号（时间序列）
    """
    
    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        image_transforms: Optional[Compose] = None,
        text_transforms: Optional[callable] = None,
    ):
        """
        初始化多模态数据集
        
        Args:
            data_list: 数据列表，每个元素包含不同模态的数据
            image_transforms: 图像变换（MONAI Compose）
            text_transforms: 文本变换函数
        """
        self.data_list = data_list
        self.image_transforms = image_transforms
        self.text_transforms = text_transforms
        logger.info(f"初始化多模态数据集，共 {len(data_list)} 条数据")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
        
        Returns:
            包含多模态数据的字典
        """
        data = self.data_list[idx]
        sample = {}
        
        # 处理医学影像
        if "image" in data:
            image = data["image"]
            if self.image_transforms:
                image = self.image_transforms(image)
            sample["image"] = image
        
        # 处理文本数据
        if "text" in data:
            text = data["text"]
            if self.text_transforms:
                text = self.text_transforms(text)
            sample["text"] = text
        
        # 处理实验室检查数据
        if "lab_data" in data:
            sample["lab_data"] = torch.tensor(data["lab_data"], dtype=torch.float32)
        
        # 处理时间序列数据
        if "timeseries" in data:
            sample["timeseries"] = torch.tensor(
                data["timeseries"], dtype=torch.float32
            )
        
        # 标签（如果存在）
        if "label" in data:
            sample["label"] = data["label"]
        
        # 元数据
        if "metadata" in data:
            sample["metadata"] = data["metadata"]
        
        return sample


class MultimodalDataLoader:
    """
    多模态数据加载器封装类
    
    提供便捷的数据加载接口，支持批处理和异步加载
    """
    
    def __init__(
        self,
        dataset: MultimodalDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        初始化数据加载器
        
        Args:
            dataset: 多模态数据集
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            pin_memory: 是否使用固定内存
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def get_dataloader(self) -> MonaiDataLoader:
        """
        获取 MONAI DataLoader
        
        Returns:
            MONAI DataLoader 实例
        """
        return MonaiDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

