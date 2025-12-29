"""pytest 配置和共享 fixtures"""
import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from src.config.settings import settings


@pytest.fixture
def temp_dir():
    """临时目录 fixture"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_data():
    """示例图像数据"""
    return np.random.rand(64, 64, 64).astype(np.float32)


@pytest.fixture
def sample_text_data():
    """示例文本数据"""
    return "患者主诉发热、咳嗽、胸痛3天。查体：体温38.5℃，双肺可闻及湿性啰音。"


@pytest.fixture
def sample_timeseries_data():
    """示例时序数据"""
    return np.random.rand(100).astype(np.float32)


@pytest.fixture
def sample_lab_data():
    """示例实验室数据"""
    return {
        "血压": 140.0,
        "血糖": 7.5,
        "白细胞": 12.0,
    }


@pytest.fixture
def sample_patient_info():
    """示例患者信息"""
    return {
        "age": 45,
        "gender": 1,  # 1: 男, 0: 女
        "bmi": 24.5,
        "allergies": [],
        "medical_history": ["高血压"],
    }


@pytest.fixture
def sample_multimodal_data(sample_image_data, sample_text_data, sample_timeseries_data):
    """示例多模态数据"""
    return {
        "image": sample_image_data,
        "text": sample_text_data,
        "timeseries": sample_timeseries_data,
    }

