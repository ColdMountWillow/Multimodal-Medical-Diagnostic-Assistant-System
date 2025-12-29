"""可解释性模块"""
from typing import Dict, List, Optional
import numpy as np
import torch

from src.utils.logger import logger


class ExplanationGenerator:
    """
    诊断解释生成器
    
    为诊断结果生成可解释的依据和推理路径
    """
    
    def __init__(self):
        """初始化解释生成器"""
        logger.info("解释生成器已初始化")
    
    def generate_explanation(
        self,
        diagnosis_result: Dict[str, any],
        patient_data: Dict[str, any],
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, any]:
        """
        生成诊断解释
        
        Args:
            diagnosis_result: 诊断结果
            patient_data: 患者数据
            feature_importance: 特征重要性（可选）
        
        Returns:
            解释字典
        """
        explanation = {
            "diagnosis": diagnosis_result.get("disease", "未知"),
            "confidence": diagnosis_result.get("confidence", 0.0),
            "key_findings": self._extract_key_findings(
                diagnosis_result, patient_data
            ),
            "reasoning": self._generate_reasoning(
                diagnosis_result, patient_data
            ),
            "supporting_evidence": self._collect_evidence(
                diagnosis_result, patient_data
            ),
        }
        
        if feature_importance:
            explanation["feature_importance"] = feature_importance
        
        return explanation
    
    def _extract_key_findings(
        self,
        diagnosis_result: Dict[str, any],
        patient_data: Dict[str, any],
    ) -> List[Dict[str, any]]:
        """
        提取关键发现
        
        Args:
            diagnosis_result: 诊断结果
            patient_data: 患者数据
        
        Returns:
            关键发现列表
        """
        findings = []
        
        # 从患者数据中提取关键信息
        if "symptoms" in patient_data:
            findings.extend([
                {"type": "症状", "value": symptom, "relevance": "high"}
                for symptom in patient_data["symptoms"]
            ])
        
        if "lab_values" in patient_data:
            abnormal_labs = {
                k: v
                for k, v in patient_data["lab_values"].items()
                if self._is_abnormal(k, v)
            }
            findings.extend([
                {
                    "type": "实验室检查",
                    "name": lab,
                    "value": value,
                    "relevance": "high",
                }
                for lab, value in abnormal_labs.items()
            ])
        
        if "imaging_features" in patient_data:
            findings.extend([
                {
                    "type": "影像特征",
                    "value": feature,
                    "relevance": "high",
                }
                for feature in patient_data["imaging_features"]
            ])
        
        return findings
    
    def _is_abnormal(self, lab_name: str, value: float) -> bool:
        """
        判断检查值是否异常（简化实现）
        
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
        }
        
        if lab_name in normal_ranges:
            min_val, max_val = normal_ranges[lab_name]
            return value < min_val or value > max_val
        
        return False
    
    def _generate_reasoning(
        self,
        diagnosis_result: Dict[str, any],
        patient_data: Dict[str, any],
    ) -> str:
        """
        生成推理过程描述
        
        Args:
            diagnosis_result: 诊断结果
            patient_data: 患者数据
        
        Returns:
            推理过程文本
        """
        disease = diagnosis_result.get("disease", "未知疾病")
        confidence = diagnosis_result.get("confidence", 0.0)
        
        reasoning_parts = [
            f"根据患者的多模态医疗数据分析，",
            f"主要发现包括：",
        ]
        
        # 添加关键发现
        if "symptoms" in patient_data:
            symptoms = patient_data["symptoms"]
            reasoning_parts.append(
                f"症状: {', '.join(symptoms[:3])}"
            )
        
        if "lab_values" in patient_data:
            abnormal_count = sum(
                1
                for k, v in patient_data["lab_values"].items()
                if self._is_abnormal(k, v)
            )
            if abnormal_count > 0:
                reasoning_parts.append(
                    f"实验室检查发现 {abnormal_count} 项异常指标"
                )
        
        reasoning_parts.extend([
            f"综合以上信息，",
            f"诊断为 {disease} 的置信度为 {confidence:.1%}。",
        ])
        
        return " ".join(reasoning_parts)
    
    def _collect_evidence(
        self,
        diagnosis_result: Dict[str, any],
        patient_data: Dict[str, any],
    ) -> List[Dict[str, any]]:
        """
        收集支持证据
        
        Args:
            diagnosis_result: 诊断结果
            patient_data: 患者数据
        
        Returns:
            证据列表
        """
        evidence = []
        
        # 从诊断结果中提取证据
        if "evidence" in diagnosis_result:
            evidence.extend([
                {"source": "规则引擎", "content": e}
                for e in diagnosis_result["evidence"]
            ])
        
        # 从患者数据中提取证据
        if "symptoms" in patient_data:
            evidence.append({
                "source": "患者主诉",
                "content": f"症状: {', '.join(patient_data['symptoms'])}",
            })
        
        if "lab_values" in patient_data:
            evidence.append({
                "source": "实验室检查",
                "content": f"检查结果: {patient_data['lab_values']}",
            })
        
        return evidence
    
    def visualize_explanation(
        self, explanation: Dict[str, any]
    ) -> Dict[str, any]:
        """
        可视化解释（返回可视化数据）
        
        Args:
            explanation: 解释字典
        
        Returns:
            可视化数据字典
        """
        visualization = {
            "diagnosis": explanation["diagnosis"],
            "confidence": explanation["confidence"],
            "key_findings_count": len(explanation["key_findings"]),
            "evidence_count": len(explanation["supporting_evidence"]),
            "reasoning_length": len(explanation["reasoning"]),
        }
        
        if "feature_importance" in explanation:
            visualization["top_features"] = sorted(
                explanation["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        
        return visualization

