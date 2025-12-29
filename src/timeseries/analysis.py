"""时序分析模块"""
from typing import List, Dict, Optional
import numpy as np
import torch

from src.timeseries.preprocessing import TimeseriesPreprocessor
from src.timeseries.models.lstm import LSTMPredictor, GRUPredictor
from src.timeseries.models.transformer import TransformerPredictor
from src.timeseries.models.anomaly_detection import AnomalyDetector
from src.utils.logger import logger


class TimeseriesAnalyzer:
    """
    时序数据分析器
    
    提供时序数据的综合分析功能
    """
    
    def __init__(
        self,
        model_type: str = "lstm",
        window_size: int = 30,
    ):
        """
        初始化分析器
        
        Args:
            model_type: 模型类型（'lstm', 'gru', 'transformer'）
            window_size: 滑动窗口大小
        """
        self.model_type = model_type
        self.window_size = window_size
        
        # 初始化预处理器
        self.preprocessor = TimeseriesPreprocessor()
        
        # 初始化预测模型
        if model_type == "lstm":
            self.predictor = LSTMPredictor()
        elif model_type == "gru":
            self.predictor = GRUPredictor()
        elif model_type == "transformer":
            self.predictor = TransformerPredictor()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 初始化异常检测器
        self.anomaly_detector = AnomalyDetector()
        
        logger.info(f"时序分析器已初始化，模型类型: {model_type}")
    
    def analyze(
        self,
        data: np.ndarray,
        forecast_steps: int = 10,
        detect_anomalies: bool = True,
    ) -> Dict[str, any]:
        """
        综合分析时序数据
        
        Args:
            data: 时序数据
            forecast_steps: 预测步数
            detect_anomalies: 是否检测异常
        
        Returns:
            分析结果字典
        """
        # 预处理
        preprocessed = self.preprocessor.preprocess(
            data, window_size=self.window_size
        )
        
        # 预测
        if preprocessed["windows"] is not None:
            windows_tensor = torch.tensor(
                preprocessed["windows"], dtype=torch.float32
            )
            predictions = self.predictor.predict(
                windows_tensor, forecast_steps=forecast_steps
            )
        else:
            # 如果没有足够的窗口，使用整个序列
            data_tensor = torch.tensor(
                preprocessed["cleaned_data"], dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1)
            predictions = self.predictor.predict(
                data_tensor, forecast_steps=forecast_steps
            )
        
        # 异常检测
        anomaly_result = None
        if detect_anomalies:
            anomaly_result = self.anomaly_detector.detect(
                preprocessed["cleaned_data"]
            )
        
        # 趋势分析
        trend = self._analyze_trend(preprocessed["cleaned_data"])
        
        return {
            "cleaned_data": preprocessed["cleaned_data"],
            "predictions": predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions,
            "anomalies": anomaly_result,
            "trend": trend,
            "features": preprocessed["features"],
            "statistics": self._compute_statistics(preprocessed["cleaned_data"]),
        }
    
    def _analyze_trend(self, data: np.ndarray) -> Dict[str, any]:
        """
        分析趋势
        
        Args:
            data: 时序数据
        
        Returns:
            趋势分析结果
        """
        if len(data) < 2:
            return {"direction": "unknown", "strength": 0.0}
        
        # 计算线性回归斜率
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        # 判断趋势方向
        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"
        
        # 计算趋势强度（相关系数）
        correlation = np.corrcoef(x, data)[0, 1]
        strength = abs(correlation)
        
        return {
            "direction": direction,
            "strength": float(strength),
            "slope": float(slope),
        }
    
    def _compute_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        计算统计信息
        
        Args:
            data: 时序数据
        
        Returns:
            统计信息字典
        """
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
        }
    
    def predict_future(
        self, data: np.ndarray, steps: int = 10
    ) -> np.ndarray:
        """
        预测未来值
        
        Args:
            data: 历史数据
            steps: 预测步数
        
        Returns:
            预测值数组
        """
        preprocessed = self.preprocessor.preprocess(data)
        
        if preprocessed["windows"] is not None:
            windows_tensor = torch.tensor(
                preprocessed["windows"], dtype=torch.float32
            )
            predictions = self.predictor.predict(
                windows_tensor, forecast_steps=steps
            )
        else:
            data_tensor = torch.tensor(
                preprocessed["cleaned_data"], dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1)
            predictions = self.predictor.predict(
                data_tensor, forecast_steps=steps
            )
        
        if isinstance(predictions, torch.Tensor):
            return predictions.cpu().numpy().flatten()
        else:
            return predictions.flatten()

