"""知识检索模块"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.knowledge.graph import KnowledgeGraph, GraphNode
from src.nlp.models.bert_model import MedicalBERTModel
from src.utils.logger import logger


class KnowledgeRetriever:
    """
    知识检索器
    
    提供知识图谱的检索和语义搜索功能
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        初始化检索器
        
        Args:
            knowledge_graph: 知识图谱实例
        """
        self.graph = knowledge_graph
        self.bert_model = MedicalBERTModel()
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self._build_embeddings()
        logger.info("知识检索器已初始化")
    
    def _build_embeddings(self) -> None:
        """构建实体嵌入向量"""
        # 为所有实体构建嵌入
        entity_texts = [
            f"{node.name} {node.entity_type}"
            for node in self.graph.nodes.values()
        ]
        
        if entity_texts:
            embeddings = self.bert_model.get_embeddings(entity_texts)
            
            for i, (entity_id, node) in enumerate(
                self.graph.nodes.items()
            ):
                if isinstance(embeddings, torch.Tensor):
                    self.entity_embeddings[entity_id] = (
                        embeddings[i].cpu().numpy()
                    )
                else:
                    self.entity_embeddings[entity_id] = embeddings[i]
    
    def semantic_search(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
        
        Returns:
            搜索结果列表
        """
        # 获取查询的嵌入
        query_embedding = self.bert_model.get_embeddings([query])
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding[0].cpu().numpy()
        else:
            query_embedding = query_embedding[0]
        
        # 计算相似度
        similarities = []
        for entity_id, entity_embedding in self.entity_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                entity_embedding.reshape(1, -1),
            )[0][0]
            similarities.append((entity_id, similarity))
        
        # 排序并返回 top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for entity_id, similarity in similarities[:top_k]:
            node = self.graph.get_entity(entity_id)
            if node:
                results.append({
                    "entity": {
                        "id": node.entity_id,
                        "name": node.name,
                        "type": node.entity_type,
                    },
                    "similarity": float(similarity),
                })
        
        return results
    
    def retrieve_by_keyword(
        self, keyword: str, entity_type: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        关键词检索
        
        Args:
            keyword: 关键词
            entity_type: 实体类型（可选）
        
        Returns:
            检索结果列表
        """
        results = self.graph.query(
            entity_name=keyword, entity_type=entity_type
        )
        
        return results
    
    def retrieve_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        max_depth: int = 2,
    ) -> List[Dict[str, any]]:
        """
        检索相关实体
        
        Args:
            entity_id: 实体 ID
            relation_type: 关系类型（可选）
            max_depth: 最大深度
        
        Returns:
            相关实体列表
        """
        related = []
        visited = {entity_id}
        queue = [(entity_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            neighbors = self.graph.get_neighbors(current_id, relation_type)
            for neighbor, edge in neighbors:
                if neighbor.entity_id not in visited:
                    visited.add(neighbor.entity_id)
                    related.append({
                        "entity": {
                            "id": neighbor.entity_id,
                            "name": neighbor.name,
                            "type": neighbor.entity_type,
                        },
                        "relation": edge.relation_type,
                        "depth": depth + 1,
                    })
                    queue.append((neighbor.entity_id, depth + 1))
        
        return related
    
    def retrieve_disease_symptoms(
        self, disease_name: str
    ) -> List[str]:
        """
        检索疾病的症状
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            症状列表
        """
        # 查找疾病实体
        disease_nodes = self.graph.query(
            entity_name=disease_name, entity_type="disease"
        )
        
        if not disease_nodes:
            return []
        
        symptoms = []
        for disease_node in disease_nodes:
            entity_id = disease_node["entity"]["id"]
            
            # 查找症状关系
            neighbors = self.graph.get_neighbors(
                entity_id, relation_type="symptom_of"
            )
            
            for neighbor, edge in neighbors:
                if neighbor.entity_type == "symptom":
                    symptoms.append(neighbor.name)
        
        return list(set(symptoms))
    
    def retrieve_treatment_options(
        self, disease_name: str
    ) -> List[Dict[str, any]]:
        """
        检索疾病的治疗方案
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            治疗方案列表
        """
        # 查找疾病实体
        disease_nodes = self.graph.query(
            entity_name=disease_name, entity_type="disease"
        )
        
        if not disease_nodes:
            return []
        
        treatments = []
        for disease_node in disease_nodes:
            entity_id = disease_node["entity"]["id"]
            
            # 查找治疗关系
            neighbors = self.graph.get_neighbors(
                entity_id, relation_type="treats"
            )
            
            for neighbor, edge in neighbors:
                if neighbor.entity_type == "drug":
                    treatments.append({
                        "drug": neighbor.name,
                        "confidence": edge.properties.get("confidence", 0.5),
                    })
        
        return treatments

