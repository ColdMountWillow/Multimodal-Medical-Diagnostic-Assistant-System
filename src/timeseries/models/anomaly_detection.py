"""异常检测模块"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.timeseries.models.lstm import LSTMPredictor
from src.config.settings import settings
from src.utils.logger import logger


class AnomalyDetector:
    """
    时序异常检测器
    
    使用多种方法检测时序数据中的异常
    """
    
    def __init__(self, method: str = "isolation_forest"):
        """
        初始化异常检测器
        
        Args:
            method: 检测方法（'isolation_forest', 'lstm', 'statistical'）
        """
        self.method = method
        self.device = torch.device(settings.DEVICE)
        
        if method == "isolation_forest":
            self.model = IsolationForest(
                contamination=0.1, random_state=42
            )
            self.scaler = StandardScaler()
        elif method == "lstm":
            self.model = LSTMPredictor(
                input_size=1, hidden_size=64, output_size=1
            )
        else:
            self.model = None
        
        logger.info(f"异常检测器已初始化，方法: {method}")
    
    def detect(
        self,
        data: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        检测异常
        
        Args:
            data: 时序数据
            threshold: 异常阈值（用于统计方法）
        
        Returns:
            检测结果字典
        """
        if self.method == "isolation_forest":
            return self._detect_isolation_forest(data)
        elif self.method == "lstm":
            return self._detect_lstm(data)
        elif self.method == "statistical":
            return self._detect_statistical(data, threshold)
        else:
            raise ValueError(f"不支持的检测方法: {self.method}")
    
    def _detect_isolation_forest(
        self, data: np.ndarray
    ) -> Dict[str, any]:
        """
        使用 Isolation Forest 检测异常
        
        Args:
            data: 时序数据
        
        Returns:
            检测结果
        """
        # 准备特征（使用滑动窗口）
        window_size = min(10, len(data) // 4)
        if window_size < 2:
            window_size = 2
        
        features = []
        for i in range(len(data) - window_size + 1):
            window = data[i : i + window_size]
            features.append([
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
            ])
        
        features = np.array(features)
        
        # 标准化
        features_scaled = self.scaler.fit_transform(features)
        
        # 训练模型（如果未训练）
        if not hasattr(self.model, "predict"):
            self.model.fit(features_scaled)
        
        # 预测
        predictions = self.model.predict(features_scaled)
        scores = self.model.score_samples(features_scaled)
        
        # 转换为异常标签
        anomaly_labels = (predictions == -1).astype(int)
        
        # 扩展到原始数据长度
        full_labels = np.zeros(len(data))
        full_scores = np.zeros(len(data))
        
        for i, (label, score) in enumerate(zip(anomaly_labels, scores)):
            full_labels[i : i + window_size] = np.maximum(
                full_labels[i : i + window_size], label
            )
            full_scores[i : i + window_size] = np.maximum(
                full_scores[i : i + window_size], -score
            )
        
        return {
            "anomalies": full_labels.astype(bool),
            "scores": full_scores,
            "anomaly_indices": np.where(full_labels == 1)[0].tolist(),
        }
    
    def _detect_lstm(self, data: np.ndarray) -> Dict[str, any]:
        """
        使用 LSTM 预测误差检测异常
        
        Args:
            data: 时序数据
        
        Returns:
            检测结果
        """
        # TODO: 实现完整的 LSTM 异常检测
        # 1. 训练 LSTM 模型预测下一个值
        # 2. 计算预测误差
        # 3. 将误差大的点标记为异常
        
        logger.warning("LSTM 异常检测未完全实现")
        
        # 简化实现：使用统计方法
        return self._detect_statistical(data)
    
    def _detect_statistical(
        self, data: np.ndarray, threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """
        使用统计方法检测异常
        
        Args:
            data: 时序数据
            threshold: 异常阈值（Z-score）
        
        Returns:
            检测结果
        """
        if threshold is None:
            threshold = 3.0
        
        # 计算 Z-score
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / (std + 1e-8))
        
        # 标记异常
        anomalies = z_scores > threshold
        
        return {
            "anomalies": anomalies,
            "scores": z_scores,
            "anomaly_indices": np.where(anomalies)[0].tolist(),
            "threshold": threshold,
        }
    
    def detect_realtime(
        self, new_value: float, history: np.ndarray
    ) -> Dict[str, any]:
        """
        实时异常检测
        
        Args:
            new_value: 新数据点
            history: 历史数据
        
        Returns:
            检测结果
        """
        # 合并数据
        full_data = np.append(history, new_value)
        
        # 检测异常
        result = self.detect(full_data)
        
        # 只返回最后一个点的结果
        return {
            "is_anomaly": result["anomalies"][-1],
            "score": result["scores"][-1],
            "value": new_value,
        }

