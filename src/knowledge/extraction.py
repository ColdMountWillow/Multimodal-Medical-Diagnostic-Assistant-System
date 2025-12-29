"""知识抽取模块"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

from src.nlp.ner import MedicalNER
from src.nlp.relation_extraction import RelationExtractor
from src.utils.logger import logger


@dataclass
class KnowledgeEntity:
    """知识实体"""
    entity_id: str
    entity_type: str  # 'disease', 'symptom', 'drug', 'test', 'anatomy'
    name: str
    aliases: List[str]
    properties: Dict[str, any]


@dataclass
class KnowledgeRelation:
    """知识关系"""
    relation_id: str
    subject_id: str
    object_id: str
    relation_type: str  # 'causes', 'treats', 'diagnoses', 'symptom_of'
    confidence: float
    source: str


class KnowledgeExtractor:
    """
    知识抽取器
    
    从医疗文本中抽取实体和关系，构建知识图谱
    """
    
    def __init__(self):
        """初始化知识抽取器"""
        self.ner = MedicalNER()
        self.relation_extractor = RelationExtractor()
        logger.info("知识抽取器已初始化")
    
    def extract_from_text(
        self, text: str, source: str = "unknown"
    ) -> Tuple[List[KnowledgeEntity], List[KnowledgeRelation]]:
        """
        从文本中抽取知识
        
        Args:
            text: 输入文本
            source: 数据来源
        
        Returns:
            (实体列表, 关系列表)
        """
        # 命名实体识别
        entities_dict = self.ner.predict(text)
        
        # 转换为知识实体
        knowledge_entities = []
        entity_map = {}  # 用于关系抽取
        
        for i, entity_dict in enumerate(entities_dict):
            entity = KnowledgeEntity(
                entity_id=f"entity_{i}",
                entity_type=entity_dict.get("type", "unknown"),
                name=entity_dict.get("text", ""),
                aliases=[],
                properties={
                    "start": entity_dict.get("start", 0),
                    "end": entity_dict.get("end", 0),
                },
            )
            knowledge_entities.append(entity)
            entity_map[entity.entity_id] = entity
        
        # 关系抽取
        relations = self.relation_extractor.extract_relations(
            text, entities_dict
        )
        
        # 转换为知识关系
        knowledge_relations = []
        for i, relation_dict in enumerate(relations):
            # 简化实现：根据实体名称匹配
            subject_id = None
            object_id = None
            
            for entity in knowledge_entities:
                if entity.name == relation_dict["subject"]["text"]:
                    subject_id = entity.entity_id
                if entity.name == relation_dict["object"]["text"]:
                    object_id = entity.entity_id
            
            if subject_id and object_id:
                relation = KnowledgeRelation(
                    relation_id=f"relation_{i}",
                    subject_id=subject_id,
                    object_id=object_id,
                    relation_type=relation_dict.get("relation", "related_to"),
                    confidence=relation_dict.get("confidence", 0.5),
                    source=source,
                )
                knowledge_relations.append(relation)
        
        logger.info(
            f"从文本中抽取了 {len(knowledge_entities)} 个实体和 "
            f"{len(knowledge_relations)} 个关系"
        )
        
        return knowledge_entities, knowledge_relations
    
    def extract_structured_knowledge(
        self, structured_data: Dict[str, any]
    ) -> Tuple[List[KnowledgeEntity], List[KnowledgeRelation]]:
        """
        从结构化数据中抽取知识
        
        Args:
            structured_data: 结构化数据字典
        
        Returns:
            (实体列表, 关系列表)
        """
        entities = []
        relations = []
        
        # 抽取疾病实体
        if "diseases" in structured_data:
            for i, disease in enumerate(structured_data["diseases"]):
                entity = KnowledgeEntity(
                    entity_id=f"disease_{i}",
                    entity_type="disease",
                    name=disease.get("name", ""),
                    aliases=disease.get("aliases", []),
                    properties=disease.get("properties", {}),
                )
                entities.append(entity)
        
        # 抽取症状实体
        if "symptoms" in structured_data:
            for i, symptom in enumerate(structured_data["symptoms"]):
                entity = KnowledgeEntity(
                    entity_id=f"symptom_{i}",
                    entity_type="symptom",
                    name=symptom,
                    aliases=[],
                    properties={},
                )
                entities.append(entity)
        
        # 抽取关系
        if "disease_symptom_map" in structured_data:
            for disease_name, symptoms in structured_data[
                "disease_symptom_map"
            ].items():
                disease_entity = next(
                    (
                        e
                        for e in entities
                        if e.name == disease_name and e.entity_type == "disease"
                    ),
                    None,
                )
                
                if disease_entity:
                    for symptom_name in symptoms:
                        symptom_entity = next(
                            (
                                e
                                for e in entities
                                if e.name == symptom_name
                                and e.entity_type == "symptom"
                            ),
                            None,
                        )
                        
                        if symptom_entity:
                            relation = KnowledgeRelation(
                                relation_id=f"rel_{len(relations)}",
                                subject_id=symptom_entity.entity_id,
                                object_id=disease_entity.entity_id,
                                relation_type="symptom_of",
                                confidence=1.0,
                                source="structured_data",
                            )
                            relations.append(relation)
        
        return entities, relations
    
    def merge_entities(
        self, entities: List[KnowledgeEntity]
    ) -> List[KnowledgeEntity]:
        """
        合并重复实体
        
        Args:
            entities: 实体列表
        
        Returns:
            合并后的实体列表
        """
        merged = {}
        
        for entity in entities:
            # 检查是否已存在相同名称的实体
            key = entity.name.lower()
            
            if key in merged:
                # 合并别名和属性
                existing = merged[key]
                existing.aliases.extend(entity.aliases)
                existing.properties.update(entity.properties)
            else:
                merged[key] = entity
        
        return list(merged.values())

