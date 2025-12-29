"""BERT 模型封装"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertForTokenClassification,
    BertForSequenceClassification,
)

from src.config.settings import settings
from src.utils.logger import logger


class MedicalBERTModel:
    """
    医疗领域 BERT 模型封装
    
    支持中文和英文医疗文本处理
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        task: str = "classification",
        num_labels: Optional[int] = None,
    ):
        """
        初始化 BERT 模型
        
        Args:
            model_name: 模型名称或路径
            task: 任务类型（'classification' 或 'token_classification'）
            num_labels: 标签数量
        """
        self.model_name = model_name
        self.task = task
        self.device = torch.device(settings.DEVICE)
        
        # 加载 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"加载 tokenizer 失败: {e}，使用默认配置")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        
        # 加载模型
        if task == "token_classification" and num_labels:
            self.model = BertForTokenClassification.from_pretrained(
                model_name, num_labels=num_labels
            ).to(self.device)
        elif task == "classification" and num_labels:
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            ).to(self.device)
        else:
            # 基础 BERT 模型
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        self.model.eval()
        logger.info(f"BERT 模型已加载: {model_name}, 任务: {task}")
    
    def encode(
        self, texts: List[str], max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        编码文本
        
        Args:
            texts: 文本列表
            max_length: 最大长度
        
        Returns:
            编码结果字典
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # 移动到设备
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        return encoded
    
    def get_embeddings(
        self, texts: List[str], max_length: int = 512
    ) -> torch.Tensor:
        """
        获取文本嵌入
        
        Args:
            texts: 文本列表
            max_length: 最大长度
        
        Returns:
            文本嵌入张量
        """
        encoded = self.encode(texts, max_length)
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            
            # 获取 [CLS] token 的嵌入
            if hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, "logits"):
                # 对于分类模型，使用池化后的输出
                embeddings = outputs.logits
            else:
                embeddings = outputs[0][:, 0, :]
        
        return embeddings
    
    def predict(
        self, texts: List[str], max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        预测
        
        Args:
            texts: 文本列表
            max_length: 最大长度
        
        Returns:
            预测结果字典
        """
        encoded = self.encode(texts, max_length)
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                return {
                    "logits": logits,
                    "probabilities": probs,
                    "predictions": predictions,
                }
            else:
                return {"embeddings": outputs[0]}


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF 模型用于序列标注
    
    常用于命名实体识别任务
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_labels: int = 5,
        num_layers: int = 2,
    ):
        """
        初始化 BiLSTM-CRF 模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            num_labels: 标签数量
            num_layers: LSTM 层数
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        
        # CRF 层（简化实现）
        self.num_labels = num_labels
    
    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs
            labels: 真实标签（训练时）
        
        Returns:
            输出字典
        """
        # 嵌入层
        embedded = self.embedding(input_ids)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # 线性层
        emissions = self.hidden2tag(lstm_out)
        
        output = {"emissions": emissions}
        
        if labels is not None:
            # 计算损失（简化实现）
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                emissions.view(-1, self.num_labels), labels.view(-1)
            )
            output["loss"] = loss
        
        return output

