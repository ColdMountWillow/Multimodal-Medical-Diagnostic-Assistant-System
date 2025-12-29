"""治疗方案优化模块"""
from typing import Dict, List, Optional
import numpy as np

from src.treatment.recommendation import TreatmentRecommender, TreatmentOption
from src.treatment.drug_interaction import DrugInteractionChecker
from src.treatment.guideline_engine import GuidelineEngine
from src.utils.logger import logger


class TreatmentOptimizer:
    """
    治疗方案优化器
    
    优化治疗方案，考虑多个因素：有效性、安全性、成本、指南符合度
    """
    
    def __init__(self):
        """初始化优化器"""
        self.recommender = TreatmentRecommender()
        self.interaction_checker = DrugInteractionChecker()
        self.guideline_engine = GuidelineEngine()
        logger.info("治疗方案优化器已初始化")
    
    def optimize_treatment_plan(
        self,
        diagnosis: str,
        patient_info: Dict[str, any],
        initial_treatments: Optional[List[str]] = None,
        optimization_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, any]:
        """
        优化治疗方案
        
        Args:
            diagnosis: 诊断结果
            patient_info: 患者信息
            initial_treatments: 初始治疗方案（可选）
            optimization_weights: 优化权重（有效性、安全性、成本、指南符合度）
        
        Returns:
            优化后的治疗方案
        """
        if optimization_weights is None:
            optimization_weights = {
                "effectiveness": 0.4,
                "safety": 0.3,
                "cost": 0.1,
                "guideline_compliance": 0.2,
            }
        
        # 获取推荐治疗方案
        recommendations = self.recommender.recommend(
            diagnosis, patient_info, top_k=10
        )
        
        if not recommendations:
            logger.warning(f"未找到 {diagnosis} 的推荐治疗方案")
            return {"optimized_plan": [], "score": 0.0}
        
        # 如果有初始方案，验证其安全性
        if initial_treatments:
            validation = self.interaction_checker.validate_treatment_plan(
                initial_treatments, patient_info
            )
            
            if not validation["is_safe"]:
                logger.warning("初始治疗方案存在安全问题，将进行优化")
        
        # 计算每个方案的优化分数
        optimized_treatments = []
        for rec in recommendations:
            treatment_name = rec["name"]
            
            # 计算综合分数
            score = self._calculate_optimization_score(
                rec,
                diagnosis,
                patient_info,
                optimization_weights,
            )
            
            # 检查安全性
            safety_check = self.interaction_checker.check_contraindications(
                treatment_name, patient_info
            )
            
            if not safety_check:  # 无禁忌症
                optimized_treatments.append({
                    **rec,
                    "optimization_score": score,
                })
        
        # 按优化分数排序
        optimized_treatments.sort(
            key=lambda x: x["optimization_score"], reverse=True
        )
        
        # 选择最佳组合（避免相互作用）
        final_plan = self._select_optimal_combination(
            optimized_treatments, patient_info
        )
        
        return {
            "optimized_plan": final_plan,
            "optimization_score": sum(
                t["optimization_score"] for t in final_plan
            ) / len(final_plan) if final_plan else 0.0,
            "optimization_weights": optimization_weights,
        }
    
    def _calculate_optimization_score(
        self,
        treatment: Dict[str, any],
        diagnosis: str,
        patient_info: Dict[str, any],
        weights: Dict[str, float],
    ) -> float:
        """
        计算优化分数
        
        Args:
            treatment: 治疗方案字典
            diagnosis: 诊断结果
            patient_info: 患者信息
            weights: 权重字典
        
        Returns:
            优化分数
        """
        # 有效性分数
        effectiveness_score = treatment.get("effectiveness_score", 0.0)
        
        # 安全性分数
        safety_score = treatment.get("safety_score", 0.0)
        
        # 成本分数（已包含在 treatment 中）
        cost_score = treatment.get("cost_score", 0.0)
        
        # 指南符合度
        guideline_match = self.guideline_engine.match_treatment_to_guideline(
            diagnosis, treatment["name"]
        )
        guideline_score = 1.0 if guideline_match else 0.5
        
        # 加权求和
        total_score = (
            effectiveness_score * weights["effectiveness"] +
            safety_score * weights["safety"] +
            cost_score * weights["cost"] +
            guideline_score * weights["guideline_compliance"]
        )
        
        return total_score
    
    def _select_optimal_combination(
        self,
        treatments: List[Dict[str, any]],
        patient_info: Dict[str, any],
    ) -> List[Dict[str, any]]:
        """
        选择最优治疗方案组合
        
        Args:
            treatments: 治疗方案列表
            patient_info: 患者信息
        
        Returns:
            最优组合
        """
        if not treatments:
            return []
        
        # 简化实现：选择前3个无相互作用的方案
        selected = []
        drug_names = []
        
        for treatment in treatments:
            treatment_name = treatment["name"]
            
            # 检查与已选药物的相互作用
            if drug_names:
                test_list = drug_names + [treatment_name]
                interactions = self.interaction_checker.check_interactions(
                    test_list
                )
                
                # 如果有严重相互作用，跳过
                severe_interactions = [
                    i
                    for i in interactions
                    if i.severity.value in ["severe", "contraindicated"]
                ]
                if severe_interactions:
                    continue
            
            selected.append(treatment)
            drug_names.append(treatment_name)
            
            # 最多选择3个方案
            if len(selected) >= 3:
                break
        
        return selected
    
    def adjust_dosage(
        self,
        treatment: str,
        patient_info: Dict[str, any],
        base_dosage: str,
    ) -> Dict[str, any]:
        """
        调整药物剂量
        
        Args:
            treatment: 药物名称
            patient_info: 患者信息
            base_dosage: 基础剂量
        
        Returns:
            调整后的剂量信息
        """
        adjustments = []
        
        # 年龄调整
        age = patient_info.get("age", 0)
        if age > 65:
            adjustments.append("老年人可能需要减量")
        
        # 肾功能调整
        renal_function = patient_info.get("renal_function", "normal")
        if renal_function == "impaired":
            adjustments.append("肾功能不全需要减量")
        
        # 肝功能调整
        liver_function = patient_info.get("liver_function", "normal")
        if liver_function == "impaired":
            adjustments.append("肝功能不全需要减量")
        
        return {
            "base_dosage": base_dosage,
            "adjusted_dosage": base_dosage,  # 简化实现
            "adjustments": adjustments,
            "monitoring": self._get_monitoring_requirements(treatment),
        }
    
    def _get_monitoring_requirements(self, treatment: str) -> List[str]:
        """
        获取监测要求
        
        Args:
            treatment: 药物名称
        
        Returns:
            监测要求列表
        """
        # 简化的监测要求（实际应从知识库获取）
        monitoring_map = {
            "华法林": ["凝血功能", "INR"],
            "地高辛": ["血药浓度", "心电图"],
            "氨基糖苷类": ["肾功能", "听力"],
        }
        
        for drug, requirements in monitoring_map.items():
            if drug in treatment:
                return requirements
        
        return ["常规监测"]

