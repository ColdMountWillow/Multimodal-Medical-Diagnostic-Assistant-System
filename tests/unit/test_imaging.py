"""医学影像模块单元测试"""
import pytest
import numpy as np
import torch

from src.imaging.detection import LesionDetector
from src.imaging.segmentation import ImageSegmentation
from src.imaging.classification import ImageClassifier


class TestLesionDetector:
    """测试病灶检测器"""
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        detector = LesionDetector(spatial_dims=2)
        assert detector is not None
    
    def test_detector_forward(self):
        """测试检测器前向传播"""
        detector = LesionDetector(spatial_dims=2)
        image = torch.randn(1, 1, 256, 256)
        
        output = detector(image)
        assert output.shape[0] == 1


class TestImageSegmentation:
    """测试图像分割"""
    
    def test_segmentation_initialization(self):
        """测试分割器初始化"""
        segmenter = ImageSegmentation(model_type="unet", spatial_dims=2)
        assert segmenter is not None
    
    def test_segmentation_forward(self):
        """测试分割器前向传播"""
        segmenter = ImageSegmentation(model_type="unet", spatial_dims=2)
        image = torch.randn(1, 1, 256, 256)
        mask = torch.randint(0, 2, (1, 1, 256, 256))
        
        output = segmenter(image, mask)
        assert "loss" in output


class TestImageClassifier:
    """测试图像分类器"""
    
    def test_classifier_initialization(self):
        """测试分类器初始化"""
        classifier = ImageClassifier(model_type="densenet", num_classes=5)
        assert classifier is not None
    
    def test_classifier_forward(self):
        """测试分类器前向传播"""
        classifier = ImageClassifier(model_type="densenet", num_classes=5)
        image = torch.randn(1, 1, 224, 224)
        label = torch.randint(0, 5, (1,))
        
        output = classifier(image, label)
        assert "logits" in output
        assert output["logits"].shape[1] == 5

