"""特征工程模块"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

from src.utils.logger import logger


class FeatureEngineer:
    """
    特征工程器
    
    从多模态数据中提取和构建风险预测特征
    """
    
    def __init__(self, normalization_method: str = "standard"):
        """
        初始化特征工程器
        
        Args:
            normalization_method: 标准化方法（'standard', 'minmax'）
        """
        self.normalization_method = normalization_method
        
        if normalization_method == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        logger.info(f"特征工程器已初始化，标准化方法: {normalization_method}")
    
    def extract_multimodal_features(
        self,
        image_features: Optional[np.ndarray] = None,
        text_features: Optional[np.ndarray] = None,
        timeseries_features: Optional[np.ndarray] = None,
        lab_data: Optional[Dict[str, float]] = None,
        patient_info: Optional[Dict[str, any]] = None,
    ) -> np.ndarray:
        """
        从多模态数据中提取特征
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            timeseries_features: 时序特征
            lab_data: 实验室数据
            patient_info: 患者信息
        
        Returns:
            融合后的特征向量
        """
        features_list = []
        
        # 图像特征
        if image_features is not None:
            if isinstance(image_features, torch.Tensor):
                image_features = image_features.cpu().numpy()
            if len(image_features.shape) > 1:
                # 如果是多维特征，进行池化
                image_features = np.mean(image_features, axis=0)
            features_list.append(image_features.flatten())
        
        # 文本特征
        if text_features is not None:
            if isinstance(text_features, torch.Tensor):
                text_features = text_features.cpu().numpy()
            if len(text_features.shape) > 1:
                text_features = np.mean(text_features, axis=0)
            features_list.append(text_features.flatten())
        
        # 时序特征
        if timeseries_features is not None:
            if isinstance(timeseries_features, torch.Tensor):
                timeseries_features = timeseries_features.cpu().numpy()
            # 提取统计特征
            ts_stats = self._extract_timeseries_stats(timeseries_features)
            features_list.append(ts_stats)
        
        # 实验室数据特征
        if lab_data:
            lab_features = self._extract_lab_features(lab_data)
            features_list.append(lab_features)
        
        # 患者信息特征
        if patient_info:
            patient_features = self._extract_patient_features(patient_info)
            features_list.append(patient_features)
        
        # 拼接所有特征
        if features_list:
            combined_features = np.concatenate(features_list)
        else:
            raise ValueError("没有可用的特征数据")
        
        return combined_features
    
    def _extract_timeseries_stats(
        self, timeseries: np.ndarray
    ) -> np.ndarray:
        """
        提取时序统计特征
        
        Args:
            timeseries: 时序数据
        
        Returns:
            统计特征向量
        """
        if len(timeseries.shape) > 1:
            timeseries = timeseries.flatten()
        
        stats = [
            np.mean(timeseries),
            np.std(timeseries),
            np.min(timeseries),
            np.max(timeseries),
            np.median(timeseries),
            np.percentile(timeseries, 25),
            np.percentile(timeseries, 75),
        ]
        
        # 趋势特征
        if len(timeseries) > 1:
            diff = np.diff(timeseries)
            stats.extend([
                np.mean(diff),
                np.std(diff),
            ])
        
        return np.array(stats)
    
    def _extract_lab_features(
        self, lab_data: Dict[str, float]
    ) -> np.ndarray:
        """
        提取实验室检查特征
        
        Args:
            lab_data: 实验室数据字典
        
        Returns:
            特征向量
        """
        # 常见实验室指标
        common_labs = [
            "血压",
            "血糖",
            "白细胞",
            "红细胞",
            "血小板",
            "肌酐",
            "尿素",
            "总胆固醇",
            "甘油三酯",
        ]
        
        features = []
        for lab in common_labs:
            if lab in lab_data:
                features.append(lab_data[lab])
            else:
                features.append(0.0)  # 缺失值用0填充
        
        # 计算异常指标数量
        abnormal_count = sum(
            1
            for lab, value in lab_data.items()
            if self._is_abnormal_lab(lab, value)
        )
        features.append(abnormal_count)
        
        return np.array(features)
    
    def _is_abnormal_lab(self, lab_name: str, value: float) -> bool:
        """
        判断实验室检查值是否异常
        
        Args:
            lab_name: 检查名称
            value: 检查值
        
        Returns:
            是否异常
        """
        # 简化的正常值范围（实际应从知识库获取）
        normal_ranges = {
            "血压": (90, 140),
            "血糖": (3.9, 6.1),
            "白细胞": (4.0, 10.0),
            "红细胞": (4.0, 5.5),
            "血小板": (100, 300),
        }
        
        if lab_name in normal_ranges:
            min_val, max_val = normal_ranges[lab_name]
            return value < min_val or value > max_val
        
        return False
    
    def _extract_patient_features(
        self, patient_info: Dict[str, any]
    ) -> np.ndarray:
        """
        提取患者信息特征
        
        Args:
            patient_info: 患者信息字典
        
        Returns:
            特征向量
        """
        features = []
        
        # 年龄
        age = patient_info.get("age", 0)
        features.append(age)
        
        # 性别（0: 女, 1: 男）
        gender = patient_info.get("gender", 0)
        features.append(gender)
        
        # BMI
        bmi = patient_info.get("bmi", 0)
        features.append(bmi)
        
        # 既往病史数量
        medical_history = patient_info.get("medical_history", [])
        features.append(len(medical_history))
        
        # 家族病史数量
        family_history = patient_info.get("family_history", [])
        features.append(len(family_history))
        
        return np.array(features)
    
    def create_interaction_features(
        self, features: np.ndarray, interaction_pairs: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        创建特征交互项
        
        Args:
            features: 原始特征向量
            interaction_pairs: 交互特征对列表
        
        Returns:
            包含交互项的特征向量
        """
        interaction_features = []
        
        for i, j in interaction_pairs:
            if i < len(features) and j < len(features):
                interaction_features.append(features[i] * features[j])
        
        if interaction_features:
            return np.concatenate([features, np.array(interaction_features)])
        else:
            return features
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 50,
        method: str = "f_classif",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        特征选择
        
        Args:
            X: 特征矩阵
            y: 标签
            k: 选择的特征数
            method: 选择方法
        
        Returns:
            (选择的特征, 特征索引)
        """
        if k >= X.shape[1]:
            return X, np.arange(X.shape[1])
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
        return X_selected, selected_indices
    
    def normalize_features(
        self, features: np.ndarray, fit: bool = True
    ) -> np.ndarray:
        """
        标准化特征
        
        Args:
            features: 特征矩阵
            fit: 是否拟合标准化器
        
        Returns:
            标准化后的特征
        """
        if fit:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)

