"""Transformer 时序预测模型"""
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import math

from src.config.settings import settings
from src.utils.logger import logger


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 (seq_len, batch_size, d_model)
        
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[: x.size(0), :]
        return x


class TransformerPredictor(nn.Module):
    """
    Transformer 时序预测模型
    
    使用 Transformer 架构进行时序预测
    """
    
    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        output_size: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
    ):
        """
        初始化 Transformer 模型
        
        Args:
            input_size: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer 层数
            dim_feedforward: 前馈网络维度
            output_size: 输出维度
            dropout: Dropout 比率
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_size = input_size
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 输出层
        self.output_projection = nn.Linear(d_model, output_size)
        
        self.device = torch.device(settings.DEVICE)
        self.to(self.device)
        
        logger.info(
            f"Transformer 模型已初始化: d_model={d_model}, "
            f"nhead={nhead}, num_layers={num_layers}"
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
            mask: 注意力掩码（可选）
        
        Returns:
            输出张量 (batch_size, output_size)
        """
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 转换为 (seq_len, batch_size, d_model) 用于 Transformer
        x = x.transpose(0, 1)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer_encoder(x, mask=mask)
        
        # 取最后一个时间步
        x = x[-1, :, :]  # (batch_size, d_model)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output
    
    def predict(
        self, data: torch.Tensor, forecast_steps: int = 1
    ) -> torch.Tensor:
        """
        预测未来值
        
        Args:
            data: 输入时序数据 (batch_size, seq_len, input_size)
            forecast_steps: 预测步数
        
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            
            data = data.to(self.device)
            
            if len(data.shape) == 2:
                data = data.unsqueeze(0)
            
            predictions = []
            current_input = data
            
            for _ in range(forecast_steps):
                output = self.forward(current_input)
                predictions.append(output)
                
                # 更新输入
                if current_input.shape[1] > 1:
                    current_input = torch.cat(
                        [current_input[:, 1:, :], output.unsqueeze(1)], dim=1
                    )
                else:
                    current_input = output.unsqueeze(1)
            
            return torch.cat(predictions, dim=1)

