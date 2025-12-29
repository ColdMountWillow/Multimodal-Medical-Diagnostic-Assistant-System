"""机器学习推理引擎"""
from typing import Dict, List, Optional
import torch
import numpy as np

from src.multimodal.fusion import MultimodalFusionModel
from src.config.settings import settings
from src.utils.logger import logger


class MLEngine:
    """
    机器学习诊断推理引擎
    
    使用深度学习模型进行诊断推理
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 100,
    ):
        """
        初始化 ML 引擎
        
        Args:
            model_path: 预训练模型路径
            num_classes: 疾病类别数
        """
        self.device = torch.device(settings.DEVICE)
        self.num_classes = num_classes
        
        # 初始化多模态融合模型
        self.model = MultimodalFusionModel(
            fusion_hidden_dim=256,
            num_classes=num_classes,
        ).to(self.device)
        
        # 加载预训练模型（如果提供）
        if model_path:
            self.load_model(model_path)
        
        # 疾病名称映射（示例）
        self.disease_names = [
            f"疾病_{i}" for i in range(num_classes)
        ]
        
        logger.info(f"ML 推理引擎已初始化，类别数: {num_classes}")
    
    def infer(
        self,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        timeseries_features: Optional[torch.Tensor] = None,
        lab_data: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> List[Dict[str, any]]:
        """
        使用 ML 模型进行推理
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            timeseries_features: 时序特征
            lab_data: 实验室数据
            top_k: 返回前 k 个结果
        
        Returns:
            诊断结果列表
        """
        self.model.eval()
        
        with torch.no_grad():
            # 准备输入
            if image_features is not None:
                if len(image_features.shape) == 2:
                    image_features = image_features.unsqueeze(0)
                image_features = image_features.to(self.device)
            
            if text_features is not None:
                if isinstance(text_features, dict):
                    text_features = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in text_features.items()
                    }
                else:
                    if len(text_features.shape) == 2:
                        text_features = text_features.unsqueeze(0)
                    text_features = text_features.to(self.device)
            
            if timeseries_features is not None:
                if len(timeseries_features.shape) == 2:
                    timeseries_features = timeseries_features.unsqueeze(0)
                timeseries_features = timeseries_features.to(self.device)
            
            if lab_data is not None:
                if len(lab_data.shape) == 1:
                    lab_data = lab_data.unsqueeze(0)
                lab_data = lab_data.to(self.device)
            
            # 前向传播
            output = self.model(
                image=image_features,
                text=text_features,
                timeseries=timeseries_features,
                lab_data=lab_data,
            )
            
            # 获取预测结果
            if "probs" in output:
                probs = output["probs"][0].cpu().numpy()
            elif "logits" in output:
                probs = torch.softmax(output["logits"][0], dim=-1).cpu().numpy()
            else:
                raise ValueError("模型输出格式不正确")
            
            # 获取 top-k
            top_k_indices = np.argsort(probs)[-top_k:][::-1]
            top_k_probs = probs[top_k_indices]
            
            results = []
            for idx, prob in zip(top_k_indices, top_k_probs):
                results.append({
                    "disease": self.disease_names[idx],
                    "disease_id": int(idx),
                    "confidence": float(prob),
                    "method": "ml_model",
                })
            
            return results
    
    def compute_confidence(
        self, predictions: List[Dict[str, any]]
    ) -> float:
        """
        计算整体置信度
        
        Args:
            predictions: 预测结果列表
        
        Returns:
            整体置信度
        """
        if not predictions:
            return 0.0
        
        # 使用最高置信度
        max_confidence = max(p["confidence"] for p in predictions)
        
        # 或者使用加权平均（如果有多个高置信度结果）
        if len(predictions) > 1:
            top2_confidence = predictions[0]["confidence"] * 0.7 + \
                             predictions[1]["confidence"] * 0.3
            return float(top2_confidence)
        
        return float(max_confidence)
    
    def load_model(self, model_path: str) -> None:
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"模型已从 {model_path} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
    
    def set_disease_names(self, disease_names: List[str]) -> None:
        """
        设置疾病名称列表
        
        Args:
            disease_names: 疾病名称列表
        """
        if len(disease_names) != self.num_classes:
            logger.warning(
                f"疾病名称数量 ({len(disease_names)}) "
                f"与类别数 ({self.num_classes}) 不匹配"
            )
        
        self.disease_names = disease_names

