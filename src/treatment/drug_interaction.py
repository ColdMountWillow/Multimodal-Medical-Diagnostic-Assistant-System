"""药物相互作用检查模块"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import logger


class InteractionSeverity(Enum):
    """相互作用严重程度"""
    MILD = "mild"  # 轻度
    MODERATE = "moderate"  # 中度
    SEVERE = "severe"  # 重度
    CONTRAINDICATED = "contraindicated"  # 禁忌


@dataclass
class DrugInteraction:
    """药物相互作用"""
    drug1: str
    drug2: str
    interaction_type: str  # 'pharmacokinetic', 'pharmacodynamic', 'pharmaceutical'
    severity: InteractionSeverity
    description: str
    recommendation: str


class DrugInteractionChecker:
    """
    药物相互作用检查器
    
    检查药物之间的相互作用和禁忌症
    """
    
    def __init__(self):
        """初始化检查器"""
        self.interaction_database = {}
        self.contraindication_database = {}
        self._load_default_interactions()
        logger.info("药物相互作用检查器已初始化")
    
    def _load_default_interactions(self) -> None:
        """加载默认相互作用数据（示例）"""
        # 示例相互作用
        self.interaction_database[("华法林", "阿司匹林")] = DrugInteraction(
            drug1="华法林",
            drug2="阿司匹林",
            interaction_type="pharmacodynamic",
            severity=InteractionSeverity.SEVERE,
            description="两者都具有抗凝作用，合用会增加出血风险",
            recommendation="避免合用，如需合用需密切监测凝血功能",
        )
        
        self.interaction_database[("地高辛", "胺碘酮")] = DrugInteraction(
            drug1="地高辛",
            drug2="胺碘酮",
            interaction_type="pharmacokinetic",
            severity=InteractionSeverity.MODERATE,
            description="胺碘酮可增加地高辛的血药浓度",
            recommendation="减少地高辛剂量，监测血药浓度",
        )
        
        # 禁忌症示例
        self.contraindication_database["阿司匹林"] = {
            "allergies": ["对阿司匹林过敏"],
            "conditions": ["活动性消化性溃疡", "严重肝肾功能不全"],
            "pregnancy": "妊娠晚期禁用",
        }
    
    def check_interactions(
        self, drug_list: List[str]
    ) -> List[DrugInteraction]:
        """
        检查药物列表中的相互作用
        
        Args:
            drug_list: 药物列表
        
        Returns:
            相互作用列表
        """
        interactions = []
        
        # 检查所有药物对
        for i, drug1 in enumerate(drug_list):
            for drug2 in drug_list[i + 1 :]:
                # 检查直接相互作用
                interaction = self._check_pair_interaction(drug1, drug2)
                if interaction:
                    interactions.append(interaction)
        
        return interactions
    
    def _check_pair_interaction(
        self, drug1: str, drug2: str
    ) -> Optional[DrugInteraction]:
        """
        检查两个药物的相互作用
        
        Args:
            drug1: 药物1
            drug2: 药物2
        
        Returns:
            相互作用对象（如果存在）
        """
        # 检查直接相互作用
        key1 = (drug1, drug2)
        key2 = (drug2, drug1)
        
        if key1 in self.interaction_database:
            return self.interaction_database[key1]
        elif key2 in self.interaction_database:
            return self.interaction_database[key2]
        
        return None
    
    def check_contraindications(
        self, drug: str, patient_info: Dict[str, any]
    ) -> List[str]:
        """
        检查药物的禁忌症
        
        Args:
            drug: 药物名称
            patient_info: 患者信息
        
        Returns:
            禁忌症列表
        """
        contraindications = []
        
        if drug not in self.contraindication_database:
            return contraindications
        
        drug_contraindications = self.contraindication_database[drug]
        
        # 检查过敏
        allergies = patient_info.get("allergies", [])
        if "allergies" in drug_contraindications:
            for allergy in drug_contraindications["allergies"]:
                if any(a in allergy for a in allergies):
                    contraindications.append(f"过敏: {allergy}")
        
        # 检查疾病禁忌
        conditions = patient_info.get("conditions", [])
        if "conditions" in drug_contraindications:
            for condition in drug_contraindications["conditions"]:
                if condition in conditions:
                    contraindications.append(f"疾病禁忌: {condition}")
        
        # 检查妊娠
        is_pregnant = patient_info.get("is_pregnant", False)
        if is_pregnant and "pregnancy" in drug_contraindications:
            contraindications.append(
                f"妊娠禁忌: {drug_contraindications['pregnancy']}"
            )
        
        return contraindications
    
    def validate_treatment_plan(
        self,
        treatment_plan: List[str],
        patient_info: Dict[str, any],
    ) -> Dict[str, any]:
        """
        验证治疗方案
        
        Args:
            treatment_plan: 治疗方案（药物列表）
            patient_info: 患者信息
        
        Returns:
            验证结果字典
        """
        # 检查相互作用
        interactions = self.check_interactions(treatment_plan)
        
        # 检查禁忌症
        contraindications = {}
        for drug in treatment_plan:
            drug_contraindications = self.check_contraindications(
                drug, patient_info
            )
            if drug_contraindications:
                contraindications[drug] = drug_contraindications
        
        # 评估安全性
        is_safe = len(interactions) == 0 and len(contraindications) == 0
        
        # 计算风险等级
        risk_level = "low"
        if interactions:
            severe_interactions = [
                i
                for i in interactions
                if i.severity == InteractionSeverity.SEVERE
                or i.severity == InteractionSeverity.CONTRAINDICATED
            ]
            if severe_interactions:
                risk_level = "high"
            else:
                risk_level = "medium"
        
        if contraindications:
            risk_level = "high"
        
        return {
            "is_safe": is_safe,
            "risk_level": risk_level,
            "interactions": [
                {
                    "drug1": i.drug1,
                    "drug2": i.drug2,
                    "severity": i.severity.value,
                    "description": i.description,
                    "recommendation": i.recommendation,
                }
                for i in interactions
            ],
            "contraindications": contraindications,
            "recommendations": self._generate_recommendations(
                interactions, contraindications
            ),
        }
    
    def _generate_recommendations(
        self,
        interactions: List[DrugInteraction],
        contraindications: Dict[str, List[str]],
    ) -> List[str]:
        """
        生成推荐建议
        
        Args:
            interactions: 相互作用列表
            contraindications: 禁忌症字典
        
        Returns:
            推荐建议列表
        """
        recommendations = []
        
        for interaction in interactions:
            if interaction.severity == InteractionSeverity.CONTRAINDICATED:
                recommendations.append(
                    f"禁止同时使用 {interaction.drug1} 和 {interaction.drug2}"
                )
            elif interaction.severity == InteractionSeverity.SEVERE:
                recommendations.append(interaction.recommendation)
        
        for drug, contra_list in contraindications.items():
            recommendations.append(
                f"{drug} 存在禁忌症，不建议使用: {', '.join(contra_list)}"
            )
        
        return recommendations
    
    def add_interaction(
        self, drug1: str, drug2: str, interaction: DrugInteraction
    ) -> None:
        """
        添加药物相互作用
        
        Args:
            drug1: 药物1
            drug2: 药物2
            interaction: 相互作用对象
        """
        self.interaction_database[(drug1, drug2)] = interaction
        logger.info(f"已添加相互作用: {drug1} <-> {drug2}")

