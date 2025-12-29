"""知识存储模块"""
from typing import Dict, List, Optional
import json
import pickle
from pathlib import Path

from src.knowledge.graph import KnowledgeGraph
from src.knowledge.extraction import KnowledgeEntity, KnowledgeRelation
from src.config.settings import settings
from src.utils.logger import logger


class KnowledgeStorage:
    """
    知识存储管理器
    
    提供知识图谱的持久化存储和加载功能
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        初始化存储管理器
        
        Args:
            storage_path: 存储路径
        """
        if storage_path is None:
            storage_path = settings.DATA_DIR / "knowledge"
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"知识存储管理器已初始化，存储路径: {storage_path}")
    
    def save_graph(
        self, graph: KnowledgeGraph, filename: str = "knowledge_graph.pkl"
    ) -> None:
        """
        保存知识图谱
        
        Args:
            graph: 知识图谱实例
            filename: 文件名
        """
        file_path = self.storage_path / filename
        
        with open(file_path, "wb") as f:
            pickle.dump(graph, f)
        
        logger.info(f"知识图谱已保存到 {file_path}")
    
    def load_graph(self, filename: str = "knowledge_graph.pkl") -> KnowledgeGraph:
        """
        加载知识图谱
        
        Args:
            filename: 文件名
        
        Returns:
            知识图谱实例
        """
        file_path = self.storage_path / filename
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}，返回空图谱")
            return KnowledgeGraph()
        
        with open(file_path, "rb") as f:
            graph = pickle.load(f)
        
        logger.info(f"知识图谱已从 {file_path} 加载")
        return graph
    
    def export_to_json(
        self, graph: KnowledgeGraph, filename: str = "knowledge_graph.json"
    ) -> None:
        """
        导出知识图谱为 JSON 格式
        
        Args:
            graph: 知识图谱实例
            filename: 文件名
        """
        file_path = self.storage_path / filename
        
        # 构建 JSON 数据结构
        json_data = {
            "nodes": [
                {
                    "id": node.entity_id,
                    "type": node.entity_type,
                    "name": node.name,
                    "properties": node.properties,
                }
                for node in graph.nodes.values()
            ],
            "edges": [
                {
                    "id": edge.relation_id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.relation_type,
                    "properties": edge.properties,
                }
                for edge in graph.edges
            ],
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"知识图谱已导出到 {file_path}")
    
    def import_from_json(
        self, filename: str = "knowledge_graph.json"
    ) -> KnowledgeGraph:
        """
        从 JSON 文件导入知识图谱
        
        Args:
            filename: 文件名
        
        Returns:
            知识图谱实例
        """
        file_path = self.storage_path / filename
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return KnowledgeGraph()
        
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        graph = KnowledgeGraph()
        
        # 导入节点
        for node_data in json_data.get("nodes", []):
            from src.knowledge.graph import GraphNode
            
            node = GraphNode(
                entity_id=node_data["id"],
                entity_type=node_data["type"],
                name=node_data["name"],
                properties=node_data.get("properties", {}),
            )
            graph.nodes[node.entity_id] = node
        
        # 导入边
        for edge_data in json_data.get("edges", []):
            from src.knowledge.graph import GraphEdge
            
            edge = GraphEdge(
                relation_id=edge_data["id"],
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                relation_type=edge_data["type"],
                properties=edge_data.get("properties", {}),
            )
            graph.edges.append(edge)
            graph.adjacency_list[edge.source_id].append(edge)
        
        logger.info(f"知识图谱已从 {file_path} 导入")
        return graph
    
    def save_entities(
        self, entities: List[KnowledgeEntity], filename: str = "entities.json"
    ) -> None:
        """
        保存实体列表
        
        Args:
            entities: 实体列表
            filename: 文件名
        """
        file_path = self.storage_path / filename
        
        entities_data = [
            {
                "id": e.entity_id,
                "type": e.entity_type,
                "name": e.name,
                "aliases": e.aliases,
                "properties": e.properties,
            }
            for e in entities
        ]
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"实体列表已保存到 {file_path}")
    
    def save_relations(
        self,
        relations: List[KnowledgeRelation],
        filename: str = "relations.json",
    ) -> None:
        """
        保存关系列表
        
        Args:
            relations: 关系列表
            filename: 文件名
        """
        file_path = self.storage_path / filename
        
        relations_data = [
            {
                "id": r.relation_id,
                "subject": r.subject_id,
                "object": r.object_id,
                "type": r.relation_type,
                "confidence": r.confidence,
                "source": r.source,
            }
            for r in relations
        ]
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(relations_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"关系列表已保存到 {file_path}")

