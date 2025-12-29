"""模型评估模块"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import logger


class RiskEvaluator:
    """
    风险预测模型评估器
    
    提供模型性能评估和可视化功能
    """
    
    def __init__(self):
        """初始化评估器"""
        logger.info("风险评估器已初始化")
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        评估分类模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选）
        
        Returns:
            评估指标字典
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        # ROC-AUC（如果有概率）
        if y_proba is not None:
            try:
                if y_proba.shape[1] > 1:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba.flatten())
            except Exception as e:
                logger.warning(f"计算 ROC-AUC 失败: {e}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # 分类报告
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
        
        return metrics
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        评估回归模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            评估指标字典
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            save_path: 保存路径（可选）
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names or range(len(np.unique(y_true))),
            yticklabels=class_names or range(len(np.unique(y_true))),
        )
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("混淆矩阵")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制 ROC 曲线
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            save_path: 保存路径（可选）
        """
        from sklearn.metrics import roc_curve
        
        if y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC 曲线 (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="随机猜测")
        plt.xlabel("假阳性率")
        plt.ylabel("真阳性率")
        plt.title("ROC 曲线")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    def evaluate_risk_stratification(
        self,
        risk_scores: np.ndarray,
        y_true: np.ndarray,
        thresholds: Dict[str, float],
    ) -> Dict[str, any]:
        """
        评估风险分层效果
        
        Args:
            risk_scores: 风险评分
            y_true: 真实标签
            thresholds: 风险阈值
        
        Returns:
            分层评估结果
        """
        # 风险分层
        risk_levels = []
        for score in risk_scores:
            if score < thresholds["low"]:
                risk_levels.append("low")
            elif score < thresholds["medium"]:
                risk_levels.append("medium")
            elif score < thresholds["high"]:
                risk_levels.append("high")
            else:
                risk_levels.append("critical")
        
        # 计算各层的实际事件率
        stratification = {}
        for level in ["low", "medium", "high", "critical"]:
            level_mask = np.array(risk_levels) == level
            if level_mask.sum() > 0:
                event_rate = y_true[level_mask].mean()
                stratification[level] = {
                    "count": int(level_mask.sum()),
                    "event_rate": float(event_rate),
                    "risk_score_mean": float(risk_scores[level_mask].mean()),
                }
        
        return {
            "stratification": stratification,
            "risk_levels": risk_levels,
        }
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        risk_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        """
        生成完整的评估报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            risk_scores: 风险评分
        
        Returns:
            评估报告字典
        """
        report = {}
        
        # 分类指标
        if y_proba is not None:
            report["classification_metrics"] = self.evaluate_classification(
                y_true, y_pred, y_proba
            )
        else:
            report["classification_metrics"] = self.evaluate_classification(
                y_true, y_pred
            )
        
        # 风险分层（如果有风险评分）
        if risk_scores is not None:
            thresholds = {"low": 0.3, "medium": 0.6, "high": 0.8}
            report["risk_stratification"] = self.evaluate_risk_stratification(
                risk_scores, y_true, thresholds
            )
        
        return report

