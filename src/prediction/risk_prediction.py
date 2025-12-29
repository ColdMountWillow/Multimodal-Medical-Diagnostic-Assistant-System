"""风险预测模型"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from lifelines import CoxPHFitter
import shap

from src.prediction.feature_engineering import FeatureEngineer
from src.config.settings import settings
from src.utils.logger import logger


class RiskPredictor:
    """
    疾病风险预测器
    
    使用多种机器学习方法进行风险预测
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        task_type: str = "classification",
        use_survival: bool = False,
    ):
        """
        初始化风险预测器
        
        Args:
            model_type: 模型类型（'xgboost', 'lightgbm', 'ensemble'）
            task_type: 任务类型（'classification', 'regression'）
            use_survival: 是否使用生存分析
        """
        self.model_type = model_type
        self.task_type = task_type
        self.use_survival = use_survival
        self.device = torch.device(settings.DEVICE)
        
        # 初始化特征工程器
        self.feature_engineer = FeatureEngineer()
        
        # 初始化模型
        if model_type == "xgboost":
            if task_type == "classification":
                self.model = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                )
            else:
                self.model = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                )
        elif model_type == "lightgbm":
            if task_type == "classification":
                self.model = LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                )
            else:
                self.model = LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                )
        elif model_type == "ensemble":
            # 集成多个模型
            self.model = None
            self.models = []
            # TODO: 实现集成模型
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 生存分析模型
        if use_survival:
            self.survival_model = CoxPHFitter()
        else:
            self.survival_model = None
        
        # SHAP 解释器
        self.shap_explainer = None
        
        logger.info(
            f"风险预测器已初始化，模型类型: {model_type}, "
            f"任务类型: {task_type}"
        )
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            feature_names: 特征名称列表
        
        Returns:
            训练指标字典
        """
        # 特征标准化
        X_normalized = self.feature_engineer.normalize_features(X, fit=True)
        
        # 训练模型
        self.model.fit(X_normalized, y)
        
        # 初始化 SHAP 解释器
        if self.model_type in ["xgboost", "lightgbm"]:
            self.shap_explainer = shap.TreeExplainer(self.model)
        
        # 计算训练指标
        train_score = self.model.score(X_normalized, y)
        
        logger.info(f"模型训练完成，训练得分: {train_score:.4f}")
        
        return {
            "train_score": train_score,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }
    
    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        预测风险
        
        Args:
            X: 特征矩阵
            return_proba: 是否返回概率
        
        Returns:
            预测结果字典
        """
        # 特征标准化
        X_normalized = self.feature_engineer.normalize_features(X, fit=False)
        
        # 预测
        if self.task_type == "classification":
            predictions = self.model.predict(X_normalized)
            result = {"predictions": predictions}
            
            if return_proba and hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X_normalized)
                result["probabilities"] = probabilities
                result["risk_scores"] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        else:
            predictions = self.model.predict(X_normalized)
            result = {
                "predictions": predictions,
                "risk_scores": predictions,
            }
        
        return result
    
    def predict_risk_level(
        self,
        risk_score: float,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        根据风险评分确定风险等级
        
        Args:
            risk_score: 风险评分（0-1）
            thresholds: 风险阈值字典
        
        Returns:
            风险等级（'low', 'medium', 'high', 'critical'）
        """
        if thresholds is None:
            thresholds = {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8,
            }
        
        if risk_score < thresholds["low"]:
            return "low"
        elif risk_score < thresholds["medium"]:
            return "medium"
        elif risk_score < thresholds["high"]:
            return "high"
        else:
            return "critical"
    
    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> Dict[str, any]:
        """
        解释预测结果（使用 SHAP）
        
        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
            top_k: 返回前 k 个重要特征
        
        Returns:
            解释结果字典
        """
        if self.shap_explainer is None:
            logger.warning("SHAP 解释器未初始化，无法生成解释")
            return {}
        
        # 特征标准化
        X_normalized = self.feature_engineer.normalize_features(X, fit=False)
        
        # 计算 SHAP 值
        shap_values = self.shap_explainer.shap_values(X_normalized)
        
        # 如果是分类任务，取正类的 SHAP 值
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # 计算平均 SHAP 值
        if len(shap_values.shape) > 1:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_shap = np.abs(shap_values)
        
        # 获取 top-k 特征
        top_k_indices = np.argsort(mean_shap)[-top_k:][::-1]
        top_k_shap = mean_shap[top_k_indices]
        
        # 构建特征重要性字典
        if feature_names:
            feature_importance = {
                feature_names[i]: float(shap)
                for i, shap in zip(top_k_indices, top_k_shap)
            }
        else:
            feature_importance = {
                f"feature_{i}": float(shap)
                for i, shap in zip(top_k_indices, top_k_shap)
            }
        
        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "top_features": list(feature_importance.keys()),
        }
    
    def predict_survival(
        self,
        X: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> pd.DataFrame:
        """
        使用生存分析预测风险
        
        Args:
            X: 特征数据框（包含 duration 和 event 列）
            duration_col: 持续时间列名
            event_col: 事件列名
        
        Returns:
            生存预测结果
        """
        if self.survival_model is None:
            raise ValueError("生存分析模型未初始化")
        
        # 训练生存模型
        self.survival_model.fit(X, duration_col=duration_col, event_col=event_col)
        
        # 预测风险
        predictions = self.survival_model.predict_partial_hazard(X)
        
        return predictions
    
    def set_risk_thresholds(
        self,
        low: float = 0.3,
        medium: float = 0.6,
        high: float = 0.8,
    ) -> None:
        """
        设置风险阈值
        
        Args:
            low: 低风险阈值
            medium: 中风险阈值
            high: 高风险阈值
        """
        self.risk_thresholds = {
            "low": low,
            "medium": medium,
            "high": high,
        }
        logger.info(f"风险阈值已设置: {self.risk_thresholds}")

