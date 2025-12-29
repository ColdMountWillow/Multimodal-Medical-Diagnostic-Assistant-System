"""文本分类模块"""
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F

from src.nlp.models.bert_model import MedicalBERTModel
from src.config.settings import settings
from src.utils.logger import logger


class TextClassifier:
    """
    医疗文本分类器
    
    对医疗文本进行分类：病历类型、疾病分类等
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        model_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        初始化文本分类器
        
        Args:
            num_classes: 分类类别数
            model_path: 预训练模型路径
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.device = torch.device(settings.DEVICE)
        
        # 使用 BERT 进行分类
        self.model = MedicalBERTModel(
            model_name=model_path or "bert-base-chinese",
            task="classification",
            num_labels=num_classes,
        )
        
        self.class_names = class_names or [
            f"类别_{i}" for i in range(num_classes)
        ]
        
        logger.info(f"文本分类器已初始化，类别数: {num_classes}")
    
    def classify(
        self, text: str, return_probabilities: bool = False
    ) -> Dict[str, any]:
        """
        对文本进行分类
        
        Args:
            text: 输入文本
            return_probabilities: 是否返回所有类别的概率
        
        Returns:
            分类结果字典
        """
        encoded = self.model.encode([text])
        
        with torch.no_grad():
            outputs = self.model.model(**encoded)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(logits, dim=-1).item()
        
        result = {
            "predicted_class": self.class_names[pred_idx],
            "class_id": pred_idx,
            "confidence": float(probs[pred_idx].item()),
        }
        
        if return_probabilities:
            result["probabilities"] = {
                name: float(prob.item())
                for name, prob in zip(self.class_names, probs)
            }
        
        return result
    
    def classify_batch(
        self, texts: List[str], return_probabilities: bool = False
    ) -> List[Dict[str, any]]:
        """
        批量分类
        
        Args:
            texts: 文本列表
            return_probabilities: 是否返回概率
        
        Returns:
            分类结果列表
        """
        results = []
        for text in texts:
            result = self.classify(text, return_probabilities)
            results.append(result)
        
        return results
    
    def get_top_k_classes(
        self, text: str, k: int = 5
    ) -> List[Dict[str, any]]:
        """
        获取 top-k 分类结果
        
        Args:
            text: 输入文本
            k: 返回前 k 个结果
        
        Returns:
            top-k 分类结果列表
        """
        encoded = self.model.encode([text])
        
        with torch.no_grad():
            outputs = self.model.model(**encoded)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=-1)
            
            top_k_probs, top_k_indices = torch.topk(probs, k)
        
        return [
            {
                "class": self.class_names[idx.item()],
                "class_id": idx.item(),
                "probability": float(prob.item()),
            }
            for prob, idx in zip(top_k_probs, top_k_indices)
        ]

