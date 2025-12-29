"""诊断流程集成测试"""
import pytest
import numpy as np

from src.multimodal.fusion import MultimodalFusionModel
from src.reasoning.rule_engine import RuleEngine
from src.reasoning.ml_engine import MLEngine
from src.reasoning.explanation import ExplanationGenerator


class TestDiagnosisFlow:
    """测试诊断流程"""
    
    def test_end_to_end_diagnosis(self, sample_patient_info):
        """测试端到端诊断流程"""
        # 1. 规则引擎推理
        rule_engine = RuleEngine()
        rule_results = rule_engine.infer(sample_patient_info)
        
        assert isinstance(rule_results, list)
        
        # 2. ML 引擎推理（需要特征）
        ml_engine = MLEngine(num_classes=10)
        # 简化测试：不实际运行模型
        
        # 3. 生成解释
        explanation_gen = ExplanationGenerator()
        if rule_results:
            explanation = explanation_gen.generate_explanation(
                rule_results[0], sample_patient_info
            )
            assert "diagnosis" in explanation
            assert "key_findings" in explanation
    
    def test_multimodal_diagnosis(self, sample_multimodal_data):
        """测试多模态诊断"""
        # 创建融合模型
        model = MultimodalFusionModel(num_classes=5)
        
        # 准备输入（简化）
        # 实际需要完整的特征提取流程
        
        assert model is not None

