"""病灶检测模块"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.inferers import SlidingWindowInferer

from src.config.settings import settings
from src.utils.logger import logger


class LesionDetector(nn.Module):
    """
    病灶检测器
    
    使用深度学习模型检测医学影像中的病灶区域
    """
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        model_path: Optional[str] = None,
    ):
        """
        初始化病灶检测器
        
        Args:
            spatial_dims: 空间维度（2 或 3）
            in_channels: 输入通道数
            out_channels: 输出通道数
            channels: UNet 各层通道数
            strides: 下采样步长
            model_path: 预训练模型路径
        """
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.device = torch.device(settings.DEVICE)
        
        # 构建 UNet 模型
        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
        ).to(self.device)
        
        # 推理器
        self.inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96) if spatial_dims == 3 else (256, 256),
            sw_batch_size=4,
            overlap=0.5,
        )
        
        # 加载预训练模型（如果提供）
        if model_path:
            self.load_model(model_path)
        
        logger.info(f"病灶检测器已初始化，空间维度: {spatial_dims}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量
        
        Returns:
            检测结果（概率图）
        """
        return self.model(x)
    
    def detect(
        self, image: torch.Tensor, threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        检测病灶
        
        Args:
            image: 输入图像
            threshold: 检测阈值
        
        Returns:
            包含检测结果的字典
        """
        self.eval()
        with torch.no_grad():
            # 确保输入在正确的设备上
            if isinstance(image, torch.Tensor):
                image = image.to(self.device)
            
            # 添加批次维度（如果需要）
            if len(image.shape) == self.spatial_dims:
                image = image.unsqueeze(0)
            
            # 推理
            if self.training:
                output = self.model(image)
            else:
                output = self.inferer(image, self.model)
            
            # 应用阈值
            binary_mask = (output > threshold).float()
            
            return {
                "probability_map": output,
                "binary_mask": binary_mask,
                "detections": self._extract_bboxes(binary_mask),
            }
    
    def _extract_bboxes(
        self, mask: torch.Tensor
    ) -> List[Dict[str, float]]:
        """
        从二值掩码中提取边界框
        
        Args:
            mask: 二值掩码
        
        Returns:
            边界框列表
        """
        # 简化实现：返回非零区域的边界框
        bboxes = []
        # 这里应该实现实际的边界框提取逻辑
        return bboxes
    
    def load_model(self, model_path: str) -> None:
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"模型已从 {model_path} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
    
    def save_model(self, model_path: str) -> None:
        """
        保存模型
        
        Args:
            model_path: 保存路径
        """
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"模型已保存到 {model_path}")

