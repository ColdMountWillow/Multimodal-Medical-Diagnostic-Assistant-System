"""时序数据预处理模块"""
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import torch
from scipy import signal
from scipy.stats import zscore

from src.utils.logger import logger


class TimeseriesPreprocessor:
    """
    时序数据预处理器
    
    提供时序数据的清洗、标准化、特征提取等功能
    """
    
    def __init__(self, normalize: bool = True):
        """
        初始化预处理器
        
        Args:
            normalize: 是否标准化数据
        """
        self.normalize = normalize
    
    def clean_data(
        self,
        data: np.ndarray,
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
    ) -> np.ndarray:
        """
        清洗数据
        
        Args:
            data: 原始时序数据
            remove_outliers: 是否移除异常值
            outlier_threshold: 异常值阈值（标准差倍数）
        
        Returns:
            清洗后的数据
        """
        cleaned_data = data.copy()
        
        if remove_outliers:
            # 使用 Z-score 方法检测异常值
            z_scores = np.abs(zscore(cleaned_data))
            cleaned_data[z_scores > outlier_threshold] = np.nan
        
        # 填充缺失值（使用前向填充）
        cleaned_data = self._fill_missing_values(cleaned_data)
        
        return cleaned_data
    
    def _fill_missing_values(self, data: np.ndarray) -> np.ndarray:
        """
        填充缺失值
        
        Args:
            data: 包含缺失值的数据
        
        Returns:
            填充后的数据
        """
        filled_data = data.copy()
        
        # 前向填充
        mask = np.isnan(filled_data)
        if mask.any():
            indices = np.where(~mask, np.arange(len(mask)), 0)
            np.maximum.accumulate(indices, out=indices)
            filled_data[mask] = filled_data[indices[mask]]
        
        # 如果开头仍有缺失值，使用后向填充
        if np.isnan(filled_data).any():
            filled_data = np.nan_to_num(filled_data, nan=np.nanmean(filled_data))
        
        return filled_data
    
    def normalize_data(
        self, data: np.ndarray, method: str = "standard"
    ) -> Tuple[np.ndarray, dict]:
        """
        标准化数据
        
        Args:
            data: 原始数据
            method: 标准化方法（'standard', 'minmax', 'robust'）
        
        Returns:
            (标准化后的数据, 标准化参数)
        """
        if method == "standard":
            mean = np.mean(data)
            std = np.std(data)
            normalized = (data - mean) / (std + 1e-8)
            params = {"mean": mean, "std": std}
        
        elif method == "minmax":
            min_val = np.min(data)
            max_val = np.max(data)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            params = {"min": min_val, "max": max_val}
        
        elif method == "robust":
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            normalized = (data - median) / (iqr + 1e-8)
            params = {"median": median, "iqr": iqr}
        
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        return normalized, params
    
    def create_sliding_windows(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        forecast_steps: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口
        
        Args:
            data: 时序数据
            window_size: 窗口大小
            stride: 滑动步长
            forecast_steps: 预测步数
        
        Returns:
            (输入窗口, 目标窗口)
        """
        X, y = [], []
        
        for i in range(0, len(data) - window_size - forecast_steps + 1, stride):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size : i + window_size + forecast_steps])
        
        return np.array(X), np.array(y)
    
    def extract_features(
        self, data: np.ndarray, window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        提取时序特征
        
        Args:
            data: 时序数据
            window_size: 窗口大小（如果为 None，对整个序列提取特征）
        
        Returns:
            特征向量
        """
        if window_size:
            # 滑动窗口特征提取
            features = []
            for i in range(0, len(data) - window_size + 1, window_size):
                window = data[i : i + window_size]
                features.append(self._extract_window_features(window))
            return np.array(features)
        else:
            # 全局特征提取
            return self._extract_window_features(data)
    
    def _extract_window_features(self, window: np.ndarray) -> np.ndarray:
        """
        提取单个窗口的特征
        
        Args:
            window: 数据窗口
        
        Returns:
            特征向量
        """
        features = []
        
        # 统计特征
        features.extend([
            np.mean(window),
            np.std(window),
            np.min(window),
            np.max(window),
            np.median(window),
        ])
        
        # 趋势特征
        if len(window) > 1:
            diff = np.diff(window)
            features.extend([
                np.mean(diff),
                np.std(diff),
            ])
        
        # 频域特征（FFT）
        if len(window) > 4:
            fft = np.fft.fft(window)
            fft_magnitude = np.abs(fft)
            features.extend([
                np.mean(fft_magnitude[: len(fft_magnitude) // 2]),
                np.max(fft_magnitude[: len(fft_magnitude) // 2]),
            ])
        
        return np.array(features)
    
    def apply_filter(
        self, data: np.ndarray, filter_type: str = "lowpass", cutoff: float = 0.1
    ) -> np.ndarray:
        """
        应用滤波器
        
        Args:
            data: 时序数据
            filter_type: 滤波器类型（'lowpass', 'highpass', 'bandpass'）
            cutoff: 截止频率
        
        Returns:
            滤波后的数据
        """
        if filter_type == "lowpass":
            b, a = signal.butter(4, cutoff, "low")
            filtered = signal.filtfilt(b, a, data)
        elif filter_type == "highpass":
            b, a = signal.butter(4, cutoff, "high")
            filtered = signal.filtfilt(b, a, data)
        else:
            raise ValueError(f"不支持的滤波器类型: {filter_type}")
        
        return filtered
    
    def preprocess(
        self,
        data: np.ndarray,
        window_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> Dict[str, any]:
        """
        完整预处理流程
        
        Args:
            data: 原始时序数据
            window_size: 窗口大小
            normalize: 是否标准化（覆盖初始化参数）
        
        Returns:
            预处理结果字典
        """
        normalize = normalize if normalize is not None else self.normalize
        
        # 清洗数据
        cleaned_data = self.clean_data(data)
        
        # 标准化
        normalization_params = None
        if normalize:
            cleaned_data, normalization_params = self.normalize_data(cleaned_data)
        
        # 创建滑动窗口（如果指定）
        windows = None
        targets = None
        if window_size:
            windows, targets = self.create_sliding_windows(
                cleaned_data, window_size
            )
        
        # 提取特征
        features = self.extract_features(cleaned_data, window_size)
        
        return {
            "cleaned_data": cleaned_data,
            "normalization_params": normalization_params,
            "windows": windows,
            "targets": targets,
            "features": features,
        }

