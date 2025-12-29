"""治疗方案推荐模块"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.utils.logger import logger


@dataclass
class TreatmentOption:
    """治疗方案选项"""
    treatment_id: str
    treatment_type: str  # 'medication', 'surgery', 'therapy', 'lifestyle'
    name: str
    description: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    effectiveness_score: float = 0.0
    safety_score: float = 0.0
    cost_score: float = 0.0
    contraindications: List[str] = None
    
    def __post_init__(self):
        if self.contraindications is None:
            self.contraindications = []


class TreatmentRecommender:
    """
    治疗方案推荐器
    
    基于诊断结果和患者特征推荐个性化治疗方案
    """
    
    def __init__(self):
        """初始化推荐器"""
        self.treatment_database = {}
        self._load_default_treatments()
        logger.info("治疗方案推荐器已初始化")
    
    def _load_default_treatments(self) -> None:
        """加载默认治疗方案（示例）"""
        # 示例治疗方案
        self.treatment_database["肺炎"] = [
            TreatmentOption(
                treatment_id="pneumonia_antibiotic_1",
                treatment_type="medication",
                name="阿莫西林",
                description="广谱抗生素，适用于细菌性肺炎",
                dosage="500mg",
                frequency="每日3次",
                duration="7-10天",
                effectiveness_score=0.85,
                safety_score=0.9,
                cost_score=0.8,
            ),
            TreatmentOption(
                treatment_id="pneumonia_antibiotic_2",
                treatment_type="medication",
                name="头孢曲松",
                description="第三代头孢菌素，适用于重症肺炎",
                dosage="1g",
                frequency="每日1次",
                duration="7-14天",
                effectiveness_score=0.9,
                safety_score=0.85,
                cost_score=0.7,
            ),
        ]
        
        self.treatment_database["糖尿病"] = [
            TreatmentOption(
                treatment_id="diabetes_metformin",
                treatment_type="medication",
                name="二甲双胍",
                description="一线降糖药物",
                dosage="500mg",
                frequency="每日2次",
                duration="长期",
                effectiveness_score=0.8,
                safety_score=0.85,
                cost_score=0.9,
            ),
            TreatmentOption(
                treatment_id="diabetes_lifestyle",
                treatment_type="lifestyle",
                name="饮食控制和运动",
                description="基础治疗方案",
                effectiveness_score=0.7,
                safety_score=1.0,
                cost_score=1.0,
            ),
        ]
    
    def recommend(
        self,
        diagnosis: str,
        patient_info: Dict[str, any],
        top_k: int = 5,
        consider_interactions: bool = True,
    ) -> List[Dict[str, any]]:
        """
        推荐治疗方案
        
        Args:
            diagnosis: 诊断结果
            patient_info: 患者信息
            top_k: 返回前 k 个推荐
            consider_interactions: 是否考虑药物相互作用
        
        Returns:
            推荐治疗方案列表
        """
        # 获取基础治疗方案
        base_treatments = self.treatment_database.get(diagnosis, [])
        
        if not base_treatments:
            logger.warning(f"未找到诊断 {diagnosis} 的治疗方案")
            return []
        
        # 计算每个方案的推荐分数
        scored_treatments = []
        for treatment in base_treatments:
            score = self._calculate_recommendation_score(
                treatment, patient_info, consider_interactions
            )
            scored_treatments.append({
                "treatment": treatment,
                "score": score,
            })
        
        # 按分数排序
        scored_treatments.sort(key=lambda x: x["score"], reverse=True)
        
        # 转换为字典格式
        recommendations = []
        for item in scored_treatments[:top_k]:
            treatment = item["treatment"]
            rec_dict = {
                "treatment_id": treatment.treatment_id,
                "treatment_type": treatment.treatment_type,
                "name": treatment.name,
                "description": treatment.description,
                "dosage": treatment.dosage,
                "frequency": treatment.frequency,
                "duration": treatment.duration,
                "recommendation_score": item["score"],
                "effectiveness_score": treatment.effectiveness_score,
                "safety_score": treatment.safety_score,
                "contraindications": treatment.contraindications,
            }
            recommendations.append(rec_dict)
        
        return recommendations
    
    def _calculate_recommendation_score(
        self,
        treatment: TreatmentOption,
        patient_info: Dict[str, any],
        consider_interactions: bool,
    ) -> float:
        """
        计算推荐分数
        
        Args:
            treatment: 治疗方案
            patient_info: 患者信息
            consider_interactions: 是否考虑相互作用
        
        Returns:
            推荐分数
        """
        # 基础分数（有效性、安全性、成本）
        base_score = (
            treatment.effectiveness_score * 0.5 +
            treatment.safety_score * 0.3 +
            treatment.cost_score * 0.2
        )
        
        # 患者特征调整
        adjustment = 0.0
        
        # 年龄调整
        age = patient_info.get("age", 0)
        if age > 65 and treatment.treatment_type == "medication":
            # 老年人可能需要调整剂量
            adjustment -= 0.1
        
        # 过敏史检查
        allergies = patient_info.get("allergies", [])
        if any(allergy in treatment.name for allergy in allergies):
            # 有过敏史，不推荐
            return 0.0
        
        # 禁忌症检查
        contraindications = patient_info.get("contraindications", [])
        if any(contra in treatment.contraindications for contra in contraindications):
            return 0.0
        
        # 药物相互作用（如果需要）
        if consider_interactions and treatment.treatment_type == "medication":
            current_medications = patient_info.get("current_medications", [])
            if current_medications:
                # 简化处理：如果有其他药物，稍微降低分数
                # 实际应调用药物相互作用检查模块
                adjustment -= 0.05 * len(current_medications)
        
        final_score = base_score + adjustment
        return max(0.0, min(1.0, final_score))
    
    def add_treatment(
        self, diagnosis: str, treatment: TreatmentOption
    ) -> None:
        """
        添加治疗方案
        
        Args:
            diagnosis: 诊断名称
            treatment: 治疗方案
        """
        if diagnosis not in self.treatment_database:
            self.treatment_database[diagnosis] = []
        
        self.treatment_database[diagnosis].append(treatment)
        logger.info(f"已添加治疗方案: {treatment.name} for {diagnosis}")
    
    def get_treatment_details(self, treatment_id: str) -> Optional[Dict[str, any]]:
        """
        获取治疗方案详情
        
        Args:
            treatment_id: 治疗方案 ID
        
        Returns:
            治疗方案详情字典
        """
        for diagnosis, treatments in self.treatment_database.items():
            for treatment in treatments:
                if treatment.treatment_id == treatment_id:
                    return {
                        "treatment_id": treatment.treatment_id,
                        "treatment_type": treatment.treatment_type,
                        "name": treatment.name,
                        "description": treatment.description,
                        "dosage": treatment.dosage,
                        "frequency": treatment.frequency,
                        "duration": treatment.duration,
                        "effectiveness_score": treatment.effectiveness_score,
                        "safety_score": treatment.safety_score,
                        "cost_score": treatment.cost_score,
                        "contraindications": treatment.contraindications,
                    }
        
        return None

