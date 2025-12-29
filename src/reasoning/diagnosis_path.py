"""诊断路径生成模块"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import logger


class DiagnosisType(Enum):
    """诊断类型"""
    PRIMARY = "primary"  # 主要诊断
    DIFFERENTIAL = "differential"  # 鉴别诊断
    COMPLICATED = "complicated"  # 并发症


@dataclass
class DiagnosisNode:
    """诊断节点"""
    disease: str
    confidence: float
    evidence: List[str]
    diagnosis_type: DiagnosisType
    parent: Optional["DiagnosisNode"] = None
    children: List["DiagnosisNode"] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class DiagnosisPathGenerator:
    """
    诊断路径生成器
    
    生成诊断推理路径和鉴别诊断树
    """
    
    def __init__(self):
        """初始化诊断路径生成器"""
        logger.info("诊断路径生成器已初始化")
    
    def generate_path(
        self,
        primary_diagnosis: Dict[str, any],
        differential_diagnoses: List[Dict[str, any]],
        patient_data: Dict[str, any],
    ) -> DiagnosisNode:
        """
        生成诊断路径
        
        Args:
            primary_diagnosis: 主要诊断
            differential_diagnoses: 鉴别诊断列表
            patient_data: 患者数据
        
        Returns:
            诊断路径根节点
        """
        # 创建主要诊断节点
        root = DiagnosisNode(
            disease=primary_diagnosis.get("disease", "未知"),
            confidence=primary_diagnosis.get("confidence", 0.0),
            evidence=primary_diagnosis.get("evidence", []),
            diagnosis_type=DiagnosisType.PRIMARY,
        )
        
        # 添加鉴别诊断节点
        for diff_diag in differential_diagnoses:
            child = DiagnosisNode(
                disease=diff_diag.get("disease", "未知"),
                confidence=diff_diag.get("confidence", 0.0),
                evidence=diff_diag.get("evidence", []),
                diagnosis_type=DiagnosisType.DIFFERENTIAL,
                parent=root,
            )
            root.children.append(child)
        
        return root
    
    def path_to_dict(self, node: DiagnosisNode) -> Dict[str, any]:
        """
        将诊断路径转换为字典
        
        Args:
            node: 诊断节点
        
        Returns:
            字典格式的诊断路径
        """
        result = {
            "disease": node.disease,
            "confidence": node.confidence,
            "type": node.diagnosis_type.value,
            "evidence": node.evidence,
            "children": [],
        }
        
        for child in node.children:
            result["children"].append(self.path_to_dict(child))
        
        return result
    
    def generate_differential_diagnosis(
        self,
        symptoms: List[str],
        lab_values: Optional[Dict[str, float]] = None,
        imaging_features: Optional[List[str]] = None,
    ) -> List[Dict[str, any]]:
        """
        生成鉴别诊断列表
        
        Args:
            symptoms: 症状列表
            lab_values: 实验室检查值
            imaging_features: 影像特征
        
        Returns:
            鉴别诊断列表
        """
        # 简化的鉴别诊断逻辑（实际应从知识库查询）
        differential_diagnoses = []
        
        # 基于症状的鉴别诊断
        symptom_keywords = {
            "发热": ["感染", "炎症", "肿瘤"],
            "咳嗽": ["肺炎", "支气管炎", "哮喘"],
            "胸痛": ["心绞痛", "心肌梗死", "胸膜炎"],
        }
        
        for symptom in symptoms:
            if symptom in symptom_keywords:
                for disease in symptom_keywords[symptom]:
                    differential_diagnoses.append({
                        "disease": disease,
                        "confidence": 0.3,
                        "evidence": [f"症状: {symptom}"],
                        "reason": f"常见于 {symptom} 症状",
                    })
        
        # 去重并按置信度排序
        seen = set()
        unique_diagnoses = []
        for diag in differential_diagnoses:
            if diag["disease"] not in seen:
                seen.add(diag["disease"])
                unique_diagnoses.append(diag)
        
        unique_diagnoses.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_diagnoses[:5]  # 返回前5个
    
    def explain_path(self, node: DiagnosisNode) -> str:
        """
        解释诊断路径
        
        Args:
            node: 诊断节点
        
        Returns:
            路径解释文本
        """
        explanation_parts = []
        
        if node.diagnosis_type == DiagnosisType.PRIMARY:
            explanation_parts.append(
                f"主要诊断: {node.disease} (置信度: {node.confidence:.1%})"
            )
        else:
            explanation_parts.append(
                f"鉴别诊断: {node.disease} (置信度: {node.confidence:.1%})"
            )
        
        if node.evidence:
            explanation_parts.append(f"依据: {', '.join(node.evidence)}")
        
        if node.children:
            explanation_parts.append("鉴别诊断:")
            for child in node.children:
                explanation_parts.append(
                    f"  - {child.disease} ({child.confidence:.1%})"
                )
        
        return "\n".join(explanation_parts)
    
    def get_diagnosis_flow(
        self, diagnosis_path: DiagnosisNode
    ) -> List[Dict[str, any]]:
        """
        获取诊断流程步骤
        
        Args:
            diagnosis_path: 诊断路径
        
        Returns:
            诊断流程步骤列表
        """
        steps = []
        
        # 第一步：主要诊断
        steps.append({
            "step": 1,
            "action": "主要诊断",
            "result": diagnosis_path.disease,
            "confidence": diagnosis_path.confidence,
        })
        
        # 后续步骤：鉴别诊断
        for i, child in enumerate(diagnosis_path.children, start=2):
            steps.append({
                "step": i,
                "action": "鉴别诊断",
                "result": child.disease,
                "confidence": child.confidence,
            })
        
        return steps

