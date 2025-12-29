"""知识图谱模块"""
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from src.knowledge.extraction import KnowledgeEntity, KnowledgeRelation
from src.utils.logger import logger


@dataclass
class GraphNode:
    """图谱节点"""
    entity_id: str
    entity_type: str
    name: str
    properties: Dict[str, any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """图谱边"""
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    医疗知识图谱
    
    存储和管理医疗知识实体和关系
    """
    
    def __init__(self):
        """初始化知识图谱"""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency_list: Dict[str, List[GraphEdge]] = defaultdict(list)
        logger.info("知识图谱已初始化")
    
    def add_entity(self, entity: KnowledgeEntity) -> None:
        """
        添加实体到图谱
        
        Args:
            entity: 知识实体
        """
        node = GraphNode(
            entity_id=entity.entity_id,
            entity_type=entity.entity_type,
            name=entity.name,
            properties={
                "aliases": entity.aliases,
                **entity.properties,
            },
        )
        
        self.nodes[entity.entity_id] = node
        logger.debug(f"已添加实体: {entity.name} ({entity.entity_id})")
    
    def add_relation(self, relation: KnowledgeRelation) -> None:
        """
        添加关系到图谱
        
        Args:
            relation: 知识关系
        """
        # 检查实体是否存在
        if (
            relation.subject_id not in self.nodes
            or relation.object_id not in self.nodes
        ):
            logger.warning(
                f"关系 {relation.relation_id} 的实体不存在，跳过"
            )
            return
        
        edge = GraphEdge(
            relation_id=relation.relation_id,
            source_id=relation.subject_id,
            target_id=relation.object_id,
            relation_type=relation.relation_type,
            properties={
                "confidence": relation.confidence,
                "source": relation.source,
            },
        )
        
        self.edges.append(edge)
        self.adjacency_list[relation.subject_id].append(edge)
        
        logger.debug(
            f"已添加关系: {relation.relation_type} "
            f"({relation.subject_id} -> {relation.object_id})"
        )
    
    def get_entity(self, entity_id: str) -> Optional[GraphNode]:
        """
        获取实体
        
        Args:
            entity_id: 实体 ID
        
        Returns:
            实体节点（如果存在）
        """
        return self.nodes.get(entity_id)
    
    def get_entities_by_type(
        self, entity_type: str
    ) -> List[GraphNode]:
        """
        按类型获取实体
        
        Args:
            entity_type: 实体类型
        
        Returns:
            实体列表
        """
        return [
            node for node in self.nodes.values() if node.entity_type == entity_type
        ]
    
    def get_neighbors(
        self, entity_id: str, relation_type: Optional[str] = None
    ) -> List[Tuple[GraphNode, GraphEdge]]:
        """
        获取实体的邻居节点
        
        Args:
            entity_id: 实体 ID
            relation_type: 关系类型（可选）
        
        Returns:
            (邻居节点, 边) 元组列表
        """
        neighbors = []
        
        for edge in self.adjacency_list.get(entity_id, []):
            if relation_type is None or edge.relation_type == relation_type:
                neighbor_id = (
                    edge.target_id
                    if edge.source_id == entity_id
                    else edge.source_id
                )
                neighbor = self.nodes.get(neighbor_id)
                if neighbor:
                    neighbors.append((neighbor, edge))
        
        return neighbors
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3,
    ) -> Optional[List[GraphEdge]]:
        """
        查找两个实体之间的路径
        
        Args:
            start_id: 起始实体 ID
            end_id: 目标实体 ID
            max_depth: 最大深度
        
        Returns:
            路径（边列表）或 None
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        # 使用 BFS 查找路径
        queue = [(start_id, [])]
        visited = {start_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            if current_id == end_id:
                return path
            
            for edge in self.adjacency_list.get(current_id, []):
                next_id = (
                    edge.target_id
                    if edge.source_id == current_id
                    else edge.source_id
                )
                
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [edge]))
        
        return None
    
    def query(
        self,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """
        查询图谱
        
        Args:
            entity_name: 实体名称（可选）
            entity_type: 实体类型（可选）
            relation_type: 关系类型（可选）
        
        Returns:
            查询结果列表
        """
        results = []
        
        # 筛选节点
        candidate_nodes = list(self.nodes.values())
        
        if entity_name:
            candidate_nodes = [
                n for n in candidate_nodes if entity_name.lower() in n.name.lower()
            ]
        
        if entity_type:
            candidate_nodes = [
                n for n in candidate_nodes if n.entity_type == entity_type
            ]
        
        # 构建结果
        for node in candidate_nodes:
            neighbors = self.get_neighbors(node.entity_id, relation_type)
            
            result = {
                "entity": {
                    "id": node.entity_id,
                    "name": node.name,
                    "type": node.entity_type,
                    "properties": node.properties,
                },
                "relations": [
                    {
                        "target": {
                            "id": neighbor.entity_id,
                            "name": neighbor.name,
                            "type": neighbor.entity_type,
                        },
                        "relation_type": edge.relation_type,
                        "properties": edge.properties,
                    }
                    for neighbor, edge in neighbors
                ],
            }
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, any]:
        """
        获取图谱统计信息
        
        Returns:
            统计信息字典
        """
        entity_types = defaultdict(int)
        relation_types = defaultdict(int)
        
        for node in self.nodes.values():
            entity_types[node.entity_type] += 1
        
        for edge in self.edges:
            relation_types[edge.relation_type] += 1
        
        return {
            "total_entities": len(self.nodes),
            "total_relations": len(self.edges),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
        }

