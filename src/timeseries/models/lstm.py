"""LSTM 时序预测模型"""
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import settings
from src.utils.logger import logger


class LSTMPredictor(nn.Module):
    """
    LSTM 时序预测模型
    
    用于预测时间序列的未来值
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        初始化 LSTM 模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM 层数
            output_size: 输出维度
            dropout: Dropout 比率
            bidirectional: 是否使用双向 LSTM
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # 输出层
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        self.device = torch.device(settings.DEVICE)
        self.to(self.device)
        
        logger.info(
            f"LSTM 模型已初始化: input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}"
        )
    
    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
            hidden: 隐藏状态（可选）
        
        Returns:
            (输出, (隐藏状态, 细胞状态))
        """
        # LSTM 前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步的输出
        output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(output)
        
        return output, hidden
    
    def predict(
        self,
        data: torch.Tensor,
        forecast_steps: int = 1,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        预测未来值
        
        Args:
            data: 输入时序数据 (batch_size, seq_len, input_size)
            forecast_steps: 预测步数
            return_hidden: 是否返回隐藏状态
        
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            
            data = data.to(self.device)
            
            # 如果输入是 2D，添加 batch 维度
            if len(data.shape) == 2:
                data = data.unsqueeze(0)
            
            predictions = []
            hidden = None
            
            # 多步预测
            current_input = data
            for _ in range(forecast_steps):
                output, hidden = self.forward(current_input, hidden)
                predictions.append(output)
                
                # 更新输入（使用预测值）
                if current_input.shape[1] > 1:
                    # 滑动窗口
                    current_input = torch.cat(
                        [current_input[:, 1:, :], output.unsqueeze(1)], dim=1
                    )
                else:
                    current_input = output.unsqueeze(1)
            
            predictions = torch.cat(predictions, dim=1)
            
            if return_hidden:
                return predictions, hidden
            else:
                return predictions


class GRUPredictor(nn.Module):
    """
    GRU 时序预测模型
    
    相比 LSTM 更轻量级的时序预测模型
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        """
        初始化 GRU 模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: GRU 层数
            output_size: 输出维度
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.device = torch.device(settings.DEVICE)
        self.to(self.device)
        
        logger.info(
            f"GRU 模型已初始化: input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}"
        )
    
    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            hidden: 隐藏状态（可选）
        
        Returns:
            (输出, 隐藏状态)
        """
        gru_out, hidden = self.gru(x, hidden)
        output = self.fc(gru_out[:, -1, :])
        return output, hidden
    
    def predict(
        self, data: torch.Tensor, forecast_steps: int = 1
    ) -> torch.Tensor:
        """预测未来值"""
        self.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            
            data = data.to(self.device)
            
            if len(data.shape) == 2:
                data = data.unsqueeze(0)
            
            predictions = []
            hidden = None
            
            current_input = data
            for _ in range(forecast_steps):
                output, hidden = self.forward(current_input, hidden)
                predictions.append(output)
                
                if current_input.shape[1] > 1:
                    current_input = torch.cat(
                        [current_input[:, 1:, :], output.unsqueeze(1)], dim=1
                    )
                else:
                    current_input = output.unsqueeze(1)
            
            return torch.cat(predictions, dim=1)

