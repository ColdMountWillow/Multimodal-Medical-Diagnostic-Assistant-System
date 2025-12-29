"""文本预处理模块"""
import re
from typing import List, Dict, Optional
import jieba
import jieba.analyse

from src.utils.logger import logger


class TextPreprocessor:
    """
    医疗文本预处理器
    
    提供医疗领域特定的文本预处理功能
    """
    
    def __init__(self, use_jieba: bool = True):
        """
        初始化文本预处理器
        
        Args:
            use_jieba: 是否使用 jieba 分词
        """
        self.use_jieba = use_jieba
        if use_jieba:
            # 加载医疗词典（如果有）
            try:
                jieba.load_userdict("medical_dict.txt")
            except FileNotFoundError:
                logger.warning("未找到医疗词典文件，使用默认分词")
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
        
        Returns:
            清洗后的文本
        """
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。；：！？、]', '', text)
        
        # 移除连续的数字和字母组合（可能是ID）
        text = re.sub(r'\b[A-Z0-9]{10,}\b', '', text)
        
        return text.strip()
    
    def segment(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
        
        Returns:
            分词结果列表
        """
        if self.use_jieba:
            return list(jieba.cut(text))
        else:
            # 简单按字符分割（用于英文）
            return text.split()
    
    def extract_keywords(
        self, text: str, top_k: int = 10
    ) -> List[tuple]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回前 k 个关键词
        
        Returns:
            (关键词, 权重) 元组列表
        """
        if self.use_jieba:
            keywords = jieba.analyse.extract_tags(
                text, topK=top_k, withWeight=True
            )
            return keywords
        else:
            # 简单实现：返回高频词
            words = self.segment(text)
            word_freq = {}
            for word in words:
                if len(word) > 1:  # 过滤单字符
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(
                word_freq.items(), key=lambda x: x[1], reverse=True
            )
            return sorted_words[:top_k]
    
    def normalize_medical_terms(self, text: str) -> str:
        """
        标准化医学术语
        
        Args:
            text: 输入文本
        
        Returns:
            标准化后的文本
        """
        # 医学术语映射表（示例）
        term_mapping = {
            "CT": "计算机断层扫描",
            "MRI": "磁共振成像",
            "ECG": "心电图",
            "BP": "血压",
            "HR": "心率",
        }
        
        normalized_text = text
        for abbrev, full_term in term_mapping.items():
            normalized_text = normalized_text.replace(abbrev, full_term)
        
        return normalized_text
    
    def preprocess(
        self, text: str, normalize: bool = True
    ) -> Dict[str, any]:
        """
        完整预处理流程
        
        Args:
            text: 原始文本
            normalize: 是否标准化医学术语
        
        Returns:
            预处理结果字典
        """
        # 清洗
        cleaned_text = self.clean_text(text)
        
        # 标准化
        if normalize:
            cleaned_text = self.normalize_medical_terms(cleaned_text)
        
        # 分词
        segments = self.segment(cleaned_text)
        
        # 提取关键词
        keywords = self.extract_keywords(cleaned_text)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "segments": segments,
            "keywords": keywords,
            "length": len(cleaned_text),
            "word_count": len(segments),
        }

