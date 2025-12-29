"""关系抽取模块"""
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel

from src.nlp.models.bert_model import MedicalBERTModel
from src.config.settings import settings
from src.utils.logger import logger


class RelationExtractor:
    """
    医疗关系抽取器
    
    从医疗文本中抽取实体之间的关系
    """
    
    # 关系类型定义
    RELATION_TYPES = [
        "症状-疾病",
        "疾病-治疗",
        "疾病-检查",
        "药物-适应症",
        "药物-副作用",
        "检查-结果",
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
    ):
        """
        初始化关系抽取器
        
        Args:
            model_path: 预训练模型路径
        """
        self.device = torch.device(settings.DEVICE)
        
        # 使用 BERT 进行关系抽取
        self.model = MedicalBERTModel(
            model_name=model_path or "bert-base-chinese",
            task="classification",
            num_labels=len(self.RELATION_TYPES),
        )
        
        logger.info("关系抽取器已初始化")
    
    def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """
        抽取实体之间的关系
        
        Args:
            text: 输入文本
            entities: 实体列表
        
        Returns:
            关系列表，每个关系包含 subject, object, relation, confidence
        """
        relations = []
        
        # 遍历实体对
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1 :], start=i + 1):
                # 构建关系分类的输入
                relation_text = self._build_relation_input(
                    text, entity1, entity2
                )
                
                # 预测关系类型
                relation_type, confidence = self._predict_relation(
                    relation_text
                )
                
                if relation_type and confidence > 0.5:
                    relations.append({
                        "subject": entity1,
                        "object": entity2,
                        "relation": relation_type,
                        "confidence": float(confidence),
                    })
        
        return relations
    
    def _build_relation_input(
        self,
        text: str,
        entity1: Dict[str, any],
        entity2: Dict[str, any],
    ) -> str:
        """
        构建关系分类的输入文本
        
        Args:
            text: 原始文本
            entity1: 第一个实体
            entity2: 第二个实体
        
        Returns:
            关系分类输入文本
        """
        # 简化实现：使用实体周围的上下文
        start = min(entity1.get("start", 0), entity2.get("start", 0))
        end = max(entity1.get("end", len(text)), entity2.get("end", len(text)))
        
        # 提取实体周围的文本
        context = text[max(0, start - 50) : min(len(text), end + 50)]
        
        # 构建输入：实体1 [SEP] 实体2 [SEP] 上下文
        relation_input = f"{entity1['text']} [SEP] {entity2['text']} [SEP] {context}"
        
        return relation_input
    
    def _predict_relation(
        self, relation_text: str
    ) -> Tuple[Optional[str], float]:
        """
        预测关系类型
        
        Args:
            relation_text: 关系文本
        
        Returns:
            (关系类型, 置信度)
        """
        try:
            encoded = self.model.encode([relation_text])
            
            with torch.no_grad():
                outputs = self.model.model(**encoded)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1)
                
                max_prob, pred_idx = torch.max(probs, dim=0)
                
                if max_prob.item() > 0.5:
                    relation_type = self.RELATION_TYPES[pred_idx.item()]
                    return relation_type, max_prob.item()
                else:
                    return None, max_prob.item()
        except Exception as e:
            logger.error(f"关系预测失败: {e}")
            return None, 0.0
    
    def build_knowledge_graph(
        self, text: str, entities: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        构建知识图谱
        
        Args:
            text: 输入文本
            entities: 实体列表
        
        Returns:
            知识图谱结构（节点和边）
        """
        # 抽取关系
        relations = self.extract_relations(text, entities)
        
        # 构建图结构
        nodes = [
            {
                "id": i,
                "label": e["text"],
                "type": e["type"],
            }
            for i, e in enumerate(entities)
        ]
        
        edges = [
            {
                "source": i,
                "target": j,
                "relation": r["relation"],
                "confidence": r["confidence"],
            }
            for i, r in enumerate(relations)
            for j in [
                next(
                    (
                        idx
                        for idx, e in enumerate(entities)
                        if e == r["subject"]
                    ),
                    None,
                ),
                next(
                    (
                        idx
                        for idx, e in enumerate(entities)
                        if e == r["object"]
                    ),
                    None,
                ),
            ]
            if j is not None
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "text": text,
        }

