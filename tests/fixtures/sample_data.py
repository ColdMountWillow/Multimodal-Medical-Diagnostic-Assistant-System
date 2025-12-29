"""示例测试数据"""
import numpy as np

# 示例医学影像数据
SAMPLE_IMAGE_2D = np.random.rand(256, 256).astype(np.float32)
SAMPLE_IMAGE_3D = np.random.rand(64, 64, 64).astype(np.float32)

# 示例文本数据
SAMPLE_TEXT = "患者主诉发热、咳嗽、胸痛3天。查体：体温38.5℃，双肺可闻及湿性啰音。X线检查显示双肺下叶炎症。"

# 示例时序数据
SAMPLE_TIMESERIES = np.random.rand(100).astype(np.float32)

# 示例实验室数据
SAMPLE_LAB_DATA = {
    "血压": 140.0,
    "血糖": 7.5,
    "白细胞": 12.0,
    "红细胞": 4.5,
    "血小板": 200.0,
}

# 示例患者信息
SAMPLE_PATIENT_INFO = {
    "patient_id": "P001",
    "age": 45,
    "gender": 1,  # 1: 男, 0: 女
    "bmi": 24.5,
    "allergies": ["青霉素"],
    "medical_history": ["高血压", "糖尿病"],
    "family_history": ["心脏病"],
}

# 示例诊断数据
SAMPLE_DIAGNOSIS = {
    "disease": "肺炎",
    "confidence": 0.85,
    "evidence": ["发热", "咳嗽", "胸痛", "X线异常"],
}

