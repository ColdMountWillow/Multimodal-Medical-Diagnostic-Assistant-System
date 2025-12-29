"""多模态数据对齐"""
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np

from src.utils.logger import logger


class DataAligner:
    """
    多模态数据对齐器
    
    处理不同模态数据的时间对齐、空间对齐和特征对齐
    """
    
    def __init__(self, alignment_method: str = "temporal"):
        """
        初始化数据对齐器
        
        Args:
            alignment_method: 对齐方法（'temporal', 'spatial', 'feature'）
        """
        self.alignment_method = alignment_method
        logger.info(f"初始化数据对齐器，方法: {alignment_method}")
    
    def temporal_align(
        self,
        data_dict: Dict[str, Any],
        reference_timestamps: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        时间对齐
        
        将不同模态的数据对齐到统一的时间戳
        
        Args:
            data_dict: 包含时间序列数据的字典
            reference_timestamps: 参考时间戳列表
        
        Returns:
            对齐后的数据字典
        """
        aligned_data = {}
        
        # 如果没有参考时间戳，使用第一个有时间的模态
        if reference_timestamps is None:
            for key, value in data_dict.items():
                if isinstance(value, dict) and "timestamps" in value:
                    reference_timestamps = value["timestamps"]
                    break
        
        if reference_timestamps is None:
            logger.warning("未找到时间戳信息，跳过时间对齐")
            return data_dict
        
        # 对齐各个模态
        for key, value in data_dict.items():
            if isinstance(value, dict) and "timestamps" in value:
                aligned_data[key] = self._interpolate_to_timestamps(
                    value["data"],
                    value["timestamps"],
                    reference_timestamps,
                )
            else:
                aligned_data[key] = value
        
        return aligned_data
    
    def _interpolate_to_timestamps(
        self,
        data: np.ndarray,
        original_timestamps: List,
        target_timestamps: List,
    ) -> np.ndarray:
        """
        插值到目标时间戳
        
        Args:
            data: 原始数据
            original_timestamps: 原始时间戳
            target_timestamps: 目标时间戳
        
        Returns:
            插值后的数据
        """
        from scipy.interpolate import interp1d
        
        if len(original_timestamps) == 1:
            # 如果只有一个时间点，直接复制
            return np.tile(data, (len(target_timestamps), 1))
        
        # 线性插值
        interp_func = interp1d(
            original_timestamps,
            data,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        
        return interp_func(target_timestamps)
    
    def feature_align(
        self,
        features_dict: Dict[str, torch.Tensor],
        target_dim: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        特征对齐
        
        将不同模态的特征对齐到统一的维度
        
        Args:
            features_dict: 特征字典
            target_dim: 目标维度（如果为 None，使用最大维度）
        
        Returns:
            对齐后的特征字典
        """
        aligned_features = {}
        
        # 确定目标维度
        if target_dim is None:
            target_dim = max(
                feat.shape[-1] for feat in features_dict.values()
            )
        
        # 对齐各个特征
        for key, features in features_dict.items():
            current_dim = features.shape[-1]
            
            if current_dim < target_dim:
                # 使用线性层扩展维度
                linear = nn.Linear(current_dim, target_dim)
                aligned_features[key] = linear(features)
            elif current_dim > target_dim:
                # 使用线性层降维
                linear = nn.Linear(current_dim, target_dim)
                aligned_features[key] = linear(features)
            else:
                aligned_features[key] = features
        
        return aligned_features
    
    def align(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行对齐操作
        
        Args:
            data: 多模态数据字典
        
        Returns:
            对齐后的数据字典
        """
        if self.alignment_method == "temporal":
            return self.temporal_align(data)
        elif self.alignment_method == "feature":
            return self.feature_align(data)
        else:
            logger.warning(f"未知的对齐方法: {self.alignment_method}")
            return data

