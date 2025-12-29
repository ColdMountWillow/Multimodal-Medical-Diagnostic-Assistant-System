"""规则推理引擎"""
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from src.utils.logger import logger


@dataclass
class Rule:
    """诊断规则"""
    condition: Dict[str, any]  # 条件字典
    conclusion: str  # 结论（疾病名称）
    confidence: float  # 置信度
    evidence: List[str]  # 证据列表


class RuleEngine:
    """
    基于规则的诊断推理引擎
    
    使用医学规则进行诊断推理
    """
    
    def __init__(self):
        """初始化规则引擎"""
        self.rules: List[Rule] = []
        self._load_default_rules()
        logger.info(f"规则引擎已初始化，共 {len(self.rules)} 条规则")
    
    def _load_default_rules(self) -> None:
        """加载默认规则（示例）"""
        # 示例规则：发热 + 咳嗽 + 胸痛 -> 肺炎
        self.rules.append(Rule(
            condition={
                "symptoms": ["发热", "咳嗽", "胸痛"],
                "required_count": 2,  # 至少需要2个症状
            },
            conclusion="肺炎",
            confidence=0.7,
            evidence=["症状匹配", "常见疾病"],
        ))
        
        # 示例规则：高血压 + 高血糖 -> 糖尿病
        self.rules.append(Rule(
            condition={
                "lab_values": {
                    "血压": {"min": 140},
                    "血糖": {"min": 7.0},
                },
                "required_count": 2,
            },
            conclusion="糖尿病",
            confidence=0.8,
            evidence=["实验室检查异常", "典型指标"],
        ))
    
    def add_rule(self, rule: Rule) -> None:
        """
        添加规则
        
        Args:
            rule: 诊断规则
        """
        self.rules.append(rule)
        logger.info(f"已添加规则: {rule.conclusion}")
    
    def match_condition(
        self, condition: Dict[str, any], patient_data: Dict[str, any]
    ) -> Tuple[bool, float, List[str]]:
        """
        匹配条件
        
        Args:
            condition: 规则条件
            patient_data: 患者数据
        
        Returns:
            (是否匹配, 匹配度, 匹配的证据)
        """
        matched_evidence = []
        match_score = 0.0
        
        # 匹配症状
        if "symptoms" in condition:
            required_symptoms = condition["symptoms"]
            patient_symptoms = patient_data.get("symptoms", [])
            
            matched_symptoms = [
                s for s in required_symptoms if s in patient_symptoms
            ]
            required_count = condition.get("required_count", len(required_symptoms))
            
            if len(matched_symptoms) >= required_count:
                match_score += len(matched_symptoms) / len(required_symptoms)
                matched_evidence.extend(
                    [f"症状: {s}" for s in matched_symptoms]
                )
        
        # 匹配实验室检查值
        if "lab_values" in condition:
            required_labs = condition["lab_values"]
            patient_labs = patient_data.get("lab_values", {})
            
            matched_labs = []
            for lab_name, criteria in required_labs.items():
                if lab_name in patient_labs:
                    value = patient_labs[lab_name]
                    
                    # 检查最小值
                    if "min" in criteria and value >= criteria["min"]:
                        matched_labs.append(lab_name)
                    # 检查最大值
                    elif "max" in criteria and value <= criteria["max"]:
                        matched_labs.append(lab_name)
            
            required_count = condition.get("required_count", len(required_labs))
            if len(matched_labs) >= required_count:
                match_score += len(matched_labs) / len(required_labs)
                matched_evidence.extend(
                    [f"检查值异常: {lab}" for lab in matched_labs]
                )
        
        # 匹配影像特征
        if "imaging_features" in condition:
            required_features = condition["imaging_features"]
            patient_features = patient_data.get("imaging_features", [])
            
            matched_features = [
                f for f in required_features if f in patient_features
            ]
            if matched_features:
                match_score += len(matched_features) / len(required_features)
                matched_evidence.extend(
                    [f"影像特征: {f}" for f in matched_features]
                )
        
        is_matched = match_score > 0.5
        return is_matched, match_score, matched_evidence
    
    def infer(
        self, patient_data: Dict[str, any], top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        推理诊断
        
        Args:
            patient_data: 患者数据
            top_k: 返回前 k 个结果
        
        Returns:
            诊断结果列表
        """
        results = []
        
        for rule in self.rules:
            is_matched, match_score, evidence = self.match_condition(
                rule.condition, patient_data
            )
            
            if is_matched:
                # 计算综合置信度
                confidence = rule.confidence * match_score
                
                results.append({
                    "disease": rule.conclusion,
                    "confidence": confidence,
                    "match_score": match_score,
                    "evidence": evidence + rule.evidence,
                    "rule_id": id(rule),
                })
        
        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results[:top_k]
    
    def explain_rule(self, rule: Rule) -> str:
        """
        解释规则
        
        Args:
            rule: 诊断规则
        
        Returns:
            规则解释文本
        """
        explanation_parts = []
        
        if "symptoms" in rule.condition:
            symptoms = rule.condition["symptoms"]
            count = rule.condition.get("required_count", len(symptoms))
            explanation_parts.append(
                f"如果患者出现以下症状中的至少 {count} 个: {', '.join(symptoms)}"
            )
        
        if "lab_values" in rule.condition:
            labs = rule.condition["lab_values"]
            explanation_parts.append(
                f"实验室检查显示: {', '.join(labs.keys())} 异常"
            )
        
        explanation_parts.append(f"则可能诊断为: {rule.conclusion}")
        explanation_parts.append(f"置信度: {rule.confidence:.2f}")
        
        return "；".join(explanation_parts)

