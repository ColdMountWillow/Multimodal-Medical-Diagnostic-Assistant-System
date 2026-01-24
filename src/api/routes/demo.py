"""演示/演示数据接口"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import random
import uuid
from datetime import datetime, timedelta
import numpy as np

from src.utils.logger import logger

router = APIRouter()


class DemoRequest(BaseModel):
    """演示请求"""
    patient_name: Optional[str] = "张三"
    age: Optional[int] = 45
    gender: Optional[str] = "男"
    symptoms: Optional[List[str]] = None
    has_image: Optional[bool] = True
    has_lab_data: Optional[bool] = True


# 预定义的疾病数据
DISEASES = [
    {"name": "细菌性肺炎", "icd10": "J15.9", "severity": "moderate"},
    {"name": "病毒性肺炎", "icd10": "J12.9", "severity": "mild"},
    {"name": "慢性阻塞性肺疾病", "icd10": "J44.9", "severity": "moderate"},
    {"name": "支气管哮喘", "icd10": "J45.9", "severity": "mild"},
    {"name": "肺结核", "icd10": "A15.9", "severity": "moderate"},
]

SYMPTOMS = ["发热", "咳嗽", "胸痛", "呼吸困难", "咳痰", "乏力", "食欲不振"]

LAB_TESTS = {
    "wbc": {"name": "白细胞计数", "unit": "×10⁹/L", "normal_range": "4.0-10.0"},
    "crp": {"name": "C反应蛋白", "unit": "mg/L", "normal_range": "0-3.0"},
    "esr": {"name": "血沉", "unit": "mm/h", "normal_range": "0-15"},
    "glucose": {"name": "血糖", "unit": "mmol/L", "normal_range": "3.9-6.1"},
}


def generate_fake_image_analysis() -> Dict[str, Any]:
    """生成假的影像分析结果"""
    # 随机生成病灶位置
    lesions = []
    num_lesions = random.randint(1, 3)
    for i in range(num_lesions):
        lesions.append({
            "lesion_id": f"lesion_{i+1}",
            "location": {
                "x": random.randint(50, 200),
                "y": random.randint(50, 200),
                "z": random.randint(30, 80),
            },
            "size": {
                "width": random.randint(10, 30),
                "height": random.randint(10, 30),
                "depth": random.randint(5, 15),
            },
            "confidence": round(random.uniform(0.75, 0.95), 2),
            "type": random.choice(["阴影", "结节", "实变", "积液"]),
        })
    
    # 分类结果
    classifications = [
        {"class": "肺炎", "probability": round(random.uniform(0.7, 0.9), 2)},
        {"class": "肺结核", "probability": round(random.uniform(0.1, 0.3), 2)},
        {"class": "正常", "probability": round(random.uniform(0.0, 0.1), 2)},
    ]
    classifications.sort(key=lambda x: x["probability"], reverse=True)
    
    return {
        "detection": {
            "lesions_found": len(lesions),
            "lesions": lesions,
            "image_quality": random.choice(["良好", "一般", "优秀"]),
        },
        "segmentation": {
            "lung_volume": round(random.uniform(3000, 5000), 2),
            "lesion_volume": round(random.uniform(50, 200), 2),
            "segmentation_mask_available": True,
        },
        "classification": {
            "top_classes": classifications,
            "primary_diagnosis": classifications[0]["class"],
            "confidence": classifications[0]["probability"],
        },
    }


def generate_fake_lab_data() -> Dict[str, Any]:
    """生成假的实验室检查数据"""
    lab_results = {}
    
    # 白细胞计数（可能偏高，表示感染）
    wbc_value = round(random.uniform(8.0, 15.0), 2)
    lab_results["wbc"] = {
        "value": wbc_value,
        "name": LAB_TESTS["wbc"]["name"],
        "unit": LAB_TESTS["wbc"]["unit"],
        "normal_range": LAB_TESTS["wbc"]["normal_range"],
        "status": "elevated" if wbc_value > 10.0 else "normal",
    }
    
    # C反应蛋白（可能偏高）
    crp_value = round(random.uniform(20.0, 100.0), 2)
    lab_results["crp"] = {
        "value": crp_value,
        "name": LAB_TESTS["crp"]["name"],
        "unit": LAB_TESTS["crp"]["unit"],
        "normal_range": LAB_TESTS["crp"]["normal_range"],
        "status": "high" if crp_value > 3.0 else "normal",
    }
    
    # 血沉
    esr_value = round(random.uniform(15.0, 50.0), 2)
    lab_results["esr"] = {
        "value": esr_value,
        "name": LAB_TESTS["esr"]["name"],
        "unit": LAB_TESTS["esr"]["unit"],
        "normal_range": LAB_TESTS["esr"]["normal_range"],
        "status": "elevated" if esr_value > 15.0 else "normal",
    }
    
    return lab_results


def generate_fake_timeseries_data(days: int = 7) -> Dict[str, Any]:
    """生成假的时序数据（体温、心率等）"""
    # 生成体温数据（模拟发热过程）
    temperatures = []
    base_temp = 36.5
    for i in range(days):
        # 模拟发热曲线：先上升，后下降
        if i < 3:
            temp = base_temp + random.uniform(1.5, 2.5) - (i * 0.3)
        else:
            temp = base_temp + random.uniform(0.0, 0.5) - ((i - 3) * 0.1)
        temperatures.append(round(max(36.0, temp), 1))
    
    # 生成心率数据
    heart_rates = [random.randint(75, 100) for _ in range(days)]
    
    # 生成呼吸频率
    respiratory_rates = [random.randint(16, 24) for _ in range(days)]
    
    return {
        "temperature": {
            "values": temperatures,
            "unit": "°C",
            "trend": "decreasing",
            "current": temperatures[-1],
        },
        "heart_rate": {
            "values": heart_rates,
            "unit": "次/分",
            "trend": "stable",
            "current": heart_rates[-1],
        },
        "respiratory_rate": {
            "values": respiratory_rates,
            "unit": "次/分",
            "trend": "stable",
            "current": respiratory_rates[-1],
        },
        "timestamps": [
            (datetime.now() - timedelta(days=days-i)).isoformat()
            for i in range(days)
        ],
    }


def generate_fake_text_analysis(text: str) -> Dict[str, Any]:
    """生成假的文本分析结果"""
    entities = []
    
    # 从症状列表中提取
    for symptom in SYMPTOMS:
        if symptom in text:
            entities.append({
                "text": symptom,
                "type": "symptom",
                "confidence": round(random.uniform(0.8, 0.95), 2),
                "position": text.find(symptom),
            })
    
    # 提取数字（可能是体温、年龄等）
    import re
    numbers = re.findall(r'\d+\.?\d*', text)
    for num in numbers[:3]:  # 最多3个数字
        entities.append({
            "text": num,
            "type": "number",
            "confidence": 0.9,
        })
    
    return {
        "entities": entities,
        "keywords": random.sample(SYMPTOMS, min(5, len(SYMPTOMS))),
        "sentiment": random.choice(["中性", "轻微负面", "负面"]),
        "medical_terms_count": len(entities),
    }


@router.post("/full-analysis")
async def generate_full_demo_analysis(request: DemoRequest):
    """
    生成完整的演示分析结果
    
    接受患者信息，返回完整的假分析结果，包括：
    - 影像分析
    - 文本分析
    - 实验室检查
    - 时序数据
    - 诊断结果
    - 风险预测
    - 治疗方案
    """
    try:
        analysis_id = str(uuid.uuid4())
        
        # 随机选择一种疾病
        selected_disease = random.choice(DISEASES)
        
        # 生成症状（如果没有提供）
        if not request.symptoms:
            request.symptoms = random.sample(SYMPTOMS, random.randint(2, 4))
        
        # 生成文本内容
        text_content = f"患者{request.patient_name}，{request.age}岁，{request.gender}性。主诉：{', '.join(request.symptoms)}。"
        
        # 1. 影像分析
        image_analysis = {}
        if request.has_image:
            image_analysis = generate_fake_image_analysis()
        
        # 2. 文本分析
        text_analysis = generate_fake_text_analysis(text_content)
        
        # 3. 实验室检查
        lab_analysis = {}
        if request.has_lab_data:
            lab_analysis = generate_fake_lab_data()
        
        # 4. 时序数据
        timeseries_analysis = generate_fake_timeseries_data()
        
        # 5. 诊断结果
        diagnosis_results = [
            {
                "disease": selected_disease["name"],
                "icd10_code": selected_disease["icd10"],
                "probability": round(random.uniform(0.75, 0.92), 2),
                "confidence": "high",
                "severity": selected_disease["severity"],
                "evidence": [
                    f"影像显示{random.choice(['双肺下叶', '右肺中叶', '左肺上叶'])}可见{random.choice(['斑片状阴影', '实变影', '磨玻璃影'])}",
                    f"症状与{selected_disease['name']}高度匹配",
                    "实验室检查支持感染性病变",
                ],
            },
            {
                "disease": random.choice([d for d in DISEASES if d["name"] != selected_disease["name"]])["name"],
                "probability": round(random.uniform(0.1, 0.25), 2),
                "confidence": "low",
            },
        ]
        
        # 6. 风险预测
        risk_scores = {
            "disease_progression": round(random.uniform(0.2, 0.5), 2),
            "complication_risk": round(random.uniform(0.15, 0.35), 2),
            "mortality_risk": round(random.uniform(0.02, 0.08), 2),
            "readmission_risk": round(random.uniform(0.1, 0.25), 2),
        }
        
        # 确定风险等级
        max_risk = max(risk_scores.values())
        if max_risk < 0.3:
            risk_level = "low"
        elif max_risk < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # 7. 治疗方案
        treatments = [
            {
                "treatment_type": "medication",
                "name": random.choice(["头孢曲松", "阿莫西林", "左氧氟沙星"]),
                "dosage": random.choice(["1g", "500mg", "750mg"]),
                "frequency": random.choice(["每日1次", "每日2次", "每日3次"]),
                "duration": random.choice(["7天", "10天", "14天"]),
                "effectiveness_score": round(random.uniform(0.8, 0.95), 2),
                "safety_score": round(random.uniform(0.85, 0.95), 2),
            },
            {
                "treatment_type": "therapy",
                "name": "氧疗",
                "dosage": "2-4L/min",
                "frequency": "持续",
                "duration": "根据血氧饱和度调整",
                "effectiveness_score": 0.85,
                "safety_score": 0.95,
            },
        ]
        
        # 8. 图表数据（用于前端可视化）
        chart_data = {
            "temperature_trend": {
                "labels": timeseries_analysis["timestamps"],
                "data": timeseries_analysis["temperature"]["values"],
                "title": "体温趋势",
            },
            "lab_values": {
                "labels": list(lab_analysis.keys()) if lab_analysis else [],
                "values": [v["value"] for v in lab_analysis.values()] if lab_analysis else [],
                "normal_ranges": [v["normal_range"] for v in lab_analysis.values()] if lab_analysis else [],
                "title": "实验室检查结果",
            },
            "risk_breakdown": {
                "labels": list(risk_scores.keys()),
                "values": list(risk_scores.values()),
                "title": "风险评分分解",
            },
        }
        
        return {
            "status": "success",
            "data": {
                "analysis_id": analysis_id,
                "patient_info": {
                    "name": request.patient_name,
                    "age": request.age,
                    "gender": request.gender,
                },
                "symptoms": request.symptoms,
                "image_analysis": image_analysis,
                "text_analysis": text_analysis,
                "lab_analysis": lab_analysis,
                "timeseries_analysis": timeseries_analysis,
                "diagnosis": {
                    "primary_diagnosis": diagnosis_results[0]["disease"],
                    "all_diagnoses": diagnosis_results,
                    "confidence": diagnosis_results[0]["probability"],
                    "evidence": diagnosis_results[0].get("evidence", [
                        f"影像显示{random.choice(['双肺下叶', '右肺中叶', '左肺上叶'])}可见{random.choice(['斑片状阴影', '实变影', '磨玻璃影'])}",
                        f"症状与{selected_disease['name']}高度匹配",
                        "实验室检查支持感染性病变",
                    ]),
                    "reasoning_path": [
                        {
                            "step": 1,
                            "description": "影像分析发现肺部异常",
                            "confidence": 0.92,
                        },
                        {
                            "step": 2,
                            "description": "症状与诊断高度匹配",
                            "confidence": 0.88,
                        },
                        {
                            "step": 3,
                            "description": "实验室检查支持诊断",
                            "confidence": 0.85,
                        },
                    ],
                },
                "risk_assessment": {
                    "overall_risk": risk_level,
                    "risk_scores": risk_scores,
                    "warnings": [
                        "患者年龄较大，需密切监测",
                        "建议住院治疗",
                    ] if request.age > 60 else [],
                },
                "treatment_plan": {
                    "recommendations": treatments,
                    "duration": treatments[0]["duration"],
                    "follow_up": "治疗后3天复查影像",
                },
                "chart_data": chart_data,  # 图表数据
                "created_at": datetime.now().isoformat(),
            },
            "message": "完整分析完成",
        }
    except Exception as e:
        logger.error(f"生成演示数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@router.get("/quick-demo")
async def quick_demo():
    """
    快速演示 - 不需要输入，直接返回示例结果
    """
    request = DemoRequest(
        patient_name="示例患者",
        age=45,
        gender="男",
        symptoms=["发热", "咳嗽", "胸痛"],
    )
    return await generate_full_demo_analysis(request)


@router.post("/image-demo")
async def image_demo_analysis():
    """
    影像分析演示 - 返回假的影像分析结果
    """
    return {
        "status": "success",
        "data": {
            "analysis_id": str(uuid.uuid4()),
            **generate_fake_image_analysis(),
            "visualization": {
                "image_url": "/api/v1/demo/sample-image",  # 假的图片URL
                "annotations": [
                    {
                        "type": "bounding_box",
                        "coordinates": [100, 150, 50, 30],
                        "label": "病灶区域",
                        "confidence": 0.89,
                    }
                ],
            },
        },
        "message": "影像分析完成",
    }


@router.get("/sample-data")
async def get_sample_data():
    """
    获取示例数据 - 返回各种类型的示例数据
    """
    return {
        "status": "success",
        "data": {
            "sample_patients": [
                {
                    "name": "张三",
                    "age": 45,
                    "gender": "男",
                    "symptoms": ["发热", "咳嗽", "胸痛"],
                },
                {
                    "name": "李四",
                    "age": 62,
                    "gender": "女",
                    "symptoms": ["呼吸困难", "乏力"],
                },
            ],
            "sample_lab_data": generate_fake_lab_data(),
            "sample_timeseries": generate_fake_timeseries_data(),
            "sample_diagnoses": DISEASES,
        },
        "message": "示例数据",
    }
