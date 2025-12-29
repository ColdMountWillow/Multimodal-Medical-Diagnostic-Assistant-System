"""临床指南引擎"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import logger


@dataclass
class Guideline:
    """临床指南"""
    guideline_id: str
    disease: str
    title: str
    version: str
    publish_date: str
    recommendations: List[Dict[str, any]]
    evidence_level: str  # 'A', 'B', 'C'
    source: str


class GuidelineEngine:
    """
    临床指南引擎
    
    提供基于临床指南的治疗建议
    """
    
    def __init__(self):
        """初始化指南引擎"""
        self.guideline_database = {}
        self._load_default_guidelines()
        logger.info("临床指南引擎已初始化")
    
    def _load_default_guidelines(self) -> None:
        """加载默认临床指南（示例）"""
        # 示例指南：肺炎
        pneumonia_guideline = Guideline(
            guideline_id="pneumonia_2023",
            disease="肺炎",
            title="社区获得性肺炎诊疗指南 2023",
            version="2023.1",
            publish_date="2023-01-01",
            recommendations=[
                {
                    "category": "诊断",
                    "content": "根据临床症状、影像学检查和实验室检查进行诊断",
                    "evidence_level": "A",
                },
                {
                    "category": "治疗",
                    "content": "轻中度患者首选阿莫西林或头孢类抗生素",
                    "evidence_level": "A",
                },
                {
                    "category": "疗程",
                    "content": "一般疗程7-10天，重症患者可延长至14天",
                    "evidence_level": "B",
                },
            ],
            evidence_level="A",
            source="中华医学会呼吸病学分会",
        )
        
        self.guideline_database["肺炎"] = [pneumonia_guideline]
        
        # 示例指南：糖尿病
        diabetes_guideline = Guideline(
            guideline_id="diabetes_2023",
            disease="糖尿病",
            title="2型糖尿病诊疗指南 2023",
            version="2023.1",
            publish_date="2023-01-01",
            recommendations=[
                {
                    "category": "一线治疗",
                    "content": "首选二甲双胍，如无禁忌症",
                    "evidence_level": "A",
                },
                {
                    "category": "生活方式",
                    "content": "饮食控制和规律运动是基础治疗",
                    "evidence_level": "A",
                },
                {
                    "category": "血糖目标",
                    "content": "HbA1c < 7.0%",
                    "evidence_level": "A",
                },
            ],
            evidence_level="A",
            source="中华医学会糖尿病学分会",
        )
        
        self.guideline_database["糖尿病"] = [diabetes_guideline]
    
    def get_guidelines(
        self, disease: str, version: Optional[str] = None
    ) -> List[Guideline]:
        """
        获取疾病的临床指南
        
        Args:
            disease: 疾病名称
            version: 指南版本（可选）
        
        Returns:
            指南列表
        """
        guidelines = self.guideline_database.get(disease, [])
        
        if version:
            guidelines = [g for g in guidelines if g.version == version]
        
        # 按版本排序（最新的在前）
        guidelines.sort(key=lambda x: x.publish_date, reverse=True)
        
        return guidelines
    
    def get_recommendations(
        self,
        disease: str,
        category: Optional[str] = None,
        evidence_level: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """
        获取治疗建议
        
        Args:
            disease: 疾病名称
            category: 建议类别（可选）
            evidence_level: 证据等级（可选）
        
        Returns:
            建议列表
        """
        guidelines = self.get_guidelines(disease)
        
        all_recommendations = []
        for guideline in guidelines:
            for rec in guideline.recommendations:
                # 过滤类别
                if category and rec.get("category") != category:
                    continue
                
                # 过滤证据等级
                if evidence_level and rec.get("evidence_level") != evidence_level:
                    continue
                
                rec_with_meta = {
                    **rec,
                    "guideline_id": guideline.guideline_id,
                    "guideline_title": guideline.title,
                    "source": guideline.source,
                }
                all_recommendations.append(rec_with_meta)
        
        return all_recommendations
    
    def match_treatment_to_guideline(
        self, disease: str, treatment: str
    ) -> Optional[Dict[str, any]]:
        """
        匹配治疗方案到临床指南
        
        Args:
            disease: 疾病名称
            treatment: 治疗方案
        
        Returns:
            匹配的指南建议（如果找到）
        """
        recommendations = self.get_recommendations(disease, category="治疗")
        
        for rec in recommendations:
            if treatment.lower() in rec["content"].lower():
                return rec
        
        return None
    
    def add_guideline(self, guideline: Guideline) -> None:
        """
        添加临床指南
        
        Args:
            guideline: 指南对象
        """
        if guideline.disease not in self.guideline_database:
            self.guideline_database[guideline.disease] = []
        
        self.guideline_database[guideline.disease].append(guideline)
        logger.info(f"已添加指南: {guideline.title} for {guideline.disease}")
    
    def search_guidelines(
        self, keyword: str, disease: Optional[str] = None
    ) -> List[Guideline]:
        """
        搜索指南
        
        Args:
            keyword: 关键词
            disease: 疾病名称（可选）
        
        Returns:
            匹配的指南列表
        """
        results = []
        
        search_diseases = [disease] if disease else self.guideline_database.keys()
        
        for dis in search_diseases:
            guidelines = self.guideline_database.get(dis, [])
            for guideline in guidelines:
                if (
                    keyword.lower() in guideline.title.lower()
                    or keyword.lower() in guideline.disease.lower()
                ):
                    results.append(guideline)
        
        return results

