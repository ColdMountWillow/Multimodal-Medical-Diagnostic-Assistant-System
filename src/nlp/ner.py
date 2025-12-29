"""命名实体识别模块"""
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.nlp.models.bert_model import MedicalBERTModel, BiLSTMCRF
from src.config.settings import settings
from src.utils.logger import logger


class MedicalNER:
    """
    医疗命名实体识别器
    
    识别医疗文本中的实体：症状、疾病、药物、检查等
    """
    
    # 实体类型定义
    ENTITY_TYPES = {
        "O": 0,  # 非实体
        "B-SYMPTOM": 1,  # 症状开始
        "I-SYMPTOM": 2,  # 症状内部
        "B-DISEASE": 3,  # 疾病开始
        "I-DISEASE": 4,  # 疾病内部
        "B-DRUG": 5,  # 药物开始
        "I-DRUG": 6,  # 药物内部
        "B-TEST": 7,  # 检查开始
        "I-TEST": 8,  # 检查内部
    }
    
    def __init__(
        self,
        model_type: str = "bert",
        model_path: Optional[str] = None,
    ):
        """
        初始化 NER 模型
        
        Args:
            model_type: 模型类型（'bert' 或 'bilstm_crf'）
            model_path: 预训练模型路径
        """
        self.model_type = model_type
        self.device = torch.device(settings.DEVICE)
        
        if model_type == "bert":
            num_labels = len(self.ENTITY_TYPES)
            self.model = MedicalBERTModel(
                model_name=model_path or "bert-base-chinese",
                task="token_classification",
                num_labels=num_labels,
            )
        elif model_type == "bilstm_crf":
            # 简化实现，实际需要词汇表
            vocab_size = 10000
            self.model = BiLSTMCRF(
                vocab_size=vocab_size,
                num_labels=len(self.ENTITY_TYPES),
            ).to(self.device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        logger.info(f"NER 模型已初始化，类型: {model_type}")
    
    def predict(self, text: str) -> List[Dict[str, any]]:
        """
        识别文本中的实体
        
        Args:
            text: 输入文本
        
        Returns:
            实体列表，每个实体包含 type, text, start, end
        """
        # Tokenize
        if self.model_type == "bert":
            encoded = self.model.encode([text])
            
            with torch.no_grad():
                outputs = self.model.model(**encoded)
                predictions = torch.argmax(outputs.logits, dim=-1)[0]
            
            # 解码实体
            entities = self._decode_entities(text, predictions, encoded)
        else:
            # BiLSTM-CRF 实现
            entities = self._predict_bilstm_crf(text)
        
        return entities
    
    def _decode_entities(
        self,
        text: str,
        predictions: torch.Tensor,
        encoded: Dict[str, torch.Tensor],
    ) -> List[Dict[str, any]]:
        """
        解码实体
        
        Args:
            text: 原始文本
            predictions: 预测标签
            encoded: 编码信息
        
        Returns:
            实体列表
        """
        entities = []
        tokens = self.model.tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0]
        )
        
        current_entity = None
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            
            label_id = pred.item()
            label = list(self.ENTITY_TYPES.keys())[
                list(self.ENTITY_TYPES.values()).index(label_id)
            ]
            
            if label.startswith("B-"):
                # 新实体开始
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label.split("-")[1]
                current_entity = {
                    "type": entity_type,
                    "text": token,
                    "start": i,
                    "end": i,
                }
            elif label.startswith("I-") and current_entity:
                # 实体继续
                current_entity["text"] += token.replace("##", "")
                current_entity["end"] = i
            else:
                # 实体结束
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _predict_bilstm_crf(self, text: str) -> List[Dict[str, any]]:
        """
        使用 BiLSTM-CRF 预测（简化实现）
        
        Args:
            text: 输入文本
        
        Returns:
            实体列表
        """
        # TODO: 实现完整的 BiLSTM-CRF 预测流程
        logger.warning("BiLSTM-CRF 预测未完全实现")
        return []
    
    def extract_entities_by_type(
        self, text: str, entity_type: str
    ) -> List[str]:
        """
        按类型提取实体
        
        Args:
            text: 输入文本
            entity_type: 实体类型（'SYMPTOM', 'DISEASE', 'DRUG', 'TEST'）
        
        Returns:
            实体文本列表
        """
        entities = self.predict(text)
        return [
            e["text"]
            for e in entities
            if e["type"].upper() == entity_type.upper()
        ]

