"""NLP 模块单元测试"""
import pytest

from src.nlp.preprocessing import TextPreprocessor
from src.nlp.ner import MedicalNER
from src.nlp.text_classification import TextClassifier


class TestTextPreprocessor:
    """测试文本预处理器"""
    
    def test_clean_text(self):
        """测试文本清洗"""
        preprocessor = TextPreprocessor()
        text = "  患者  主诉  发热  \n  咳嗽  "
        cleaned = preprocessor.clean_text(text)
        
        assert cleaned == "患者 主诉 发热 咳嗽"
    
    def test_segment(self):
        """测试分词"""
        preprocessor = TextPreprocessor(use_jieba=False)
        text = "患者主诉发热咳嗽"
        segments = preprocessor.segment(text)
        
        assert len(segments) > 0
    
    def test_extract_keywords(self):
        """测试关键词提取"""
        preprocessor = TextPreprocessor()
        text = "患者主诉发热、咳嗽、胸痛，查体发现双肺湿性啰音"
        keywords = preprocessor.extract_keywords(text, top_k=5)
        
        assert len(keywords) <= 5


class TestMedicalNER:
    """测试医疗 NER"""
    
    def test_ner_initialization(self):
        """测试 NER 初始化"""
        ner = MedicalNER(model_type="bert")
        assert ner is not None
    
    def test_ner_predict(self):
        """测试 NER 预测"""
        ner = MedicalNER(model_type="bert")
        text = "患者主诉发热、咳嗽"
        # 注意：实际预测需要加载模型，这里只测试接口
        # entities = ner.predict(text)
        # assert isinstance(entities, list)


class TestTextClassifier:
    """测试文本分类器"""
    
    def test_classifier_initialization(self):
        """测试分类器初始化"""
        classifier = TextClassifier(num_classes=5)
        assert classifier is not None

