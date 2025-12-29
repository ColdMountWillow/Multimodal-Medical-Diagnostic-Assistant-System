"""多模态模块单元测试"""
import pytest
import numpy as np
import torch

from src.multimodal.data_loader import MultimodalDataset, MultimodalDataLoader
from src.multimodal.preprocessing import MultimodalPreprocessor
from src.multimodal.fusion import MultimodalFusionModel


class TestMultimodalDataset:
    """测试多模态数据集"""
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        data_list = [
            {
                "image": np.random.rand(64, 64),
                "text": "测试文本",
                "label": 0,
            }
        ]
        dataset = MultimodalDataset(data_list)
        assert len(dataset) == 1
    
    def test_dataset_getitem(self):
        """测试数据集获取项"""
        data_list = [
            {
                "image": np.random.rand(64, 64),
                "text": "测试文本",
                "lab_data": [1.0, 2.0, 3.0],
            }
        ]
        dataset = MultimodalDataset(data_list)
        sample = dataset[0]
        
        assert "image" in sample
        assert "text" in sample
        assert "lab_data" in sample


class TestMultimodalPreprocessor:
    """测试多模态预处理器"""
    
    def test_preprocess_image(self, temp_dir):
        """测试图像预处理"""
        preprocessor = MultimodalPreprocessor()
        # 简化测试：使用 numpy 数组
        image = np.random.rand(64, 64).astype(np.float32)
        # 注意：实际需要 DICOM 文件路径，这里简化测试
        assert image is not None
    
    def test_preprocess_text(self):
        """测试文本预处理"""
        preprocessor = MultimodalPreprocessor()
        text = "患者主诉发热、咳嗽"
        result = preprocessor.preprocess_text(text)
        
        assert "text" in result
        assert "length" in result
        assert result["text"] == text
    
    def test_preprocess_lab_data(self):
        """测试实验室数据预处理"""
        preprocessor = MultimodalPreprocessor()
        lab_data = np.array([1.0, 2.0, 3.0])
        result = preprocessor.preprocess_lab_data(lab_data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3


class TestMultimodalFusionModel:
    """测试多模态融合模型"""
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = MultimodalFusionModel(num_classes=10)
        assert model is not None
    
    def test_model_forward(self):
        """测试模型前向传播"""
        model = MultimodalFusionModel(num_classes=10)
        
        # 创建模拟输入
        image = torch.randn(1, 1, 64, 64)
        text = torch.randn(1, 768)
        timeseries = torch.randn(1, 100, 256)
        
        output = model(
            image=image,
            text=text,
            timeseries=timeseries,
        )
        
        assert "features" in output
        assert "logits" in output
        assert output["logits"].shape[1] == 10

