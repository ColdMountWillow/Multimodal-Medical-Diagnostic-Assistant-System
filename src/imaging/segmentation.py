"""图像分割模块"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from monai.networks.nets import UNet, SegResNet
from monai.losses import DiceLoss, FocalLoss
from monai.inferers import SlidingWindowInferer

from src.config.settings import settings
from src.utils.logger import logger


class ImageSegmentation(nn.Module):
    """
    医学图像分割器
    
    支持多种分割任务：器官分割、病灶分割等
    """
    
    def __init__(
        self,
        model_type: str = "unet",
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        model_path: Optional[str] = None,
    ):
        """
        初始化图像分割器
        
        Args:
            model_type: 模型类型（'unet' 或 'segresnet'）
            spatial_dims: 空间维度
            in_channels: 输入通道数
            out_channels: 输出通道数（分割类别数）
            model_path: 预训练模型路径
        """
        super().__init__()
        
        self.model_type = model_type
        self.spatial_dims = spatial_dims
        self.device = torch.device(settings.DEVICE)
        
        # 构建模型
        if model_type == "unet":
            self.model = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
            ).to(self.device)
        elif model_type == "segresnet":
            self.model = SegResNet(
                spatial_dims=spatial_dims,
                init_filters=16,
                in_channels=in_channels,
                out_channels=out_channels,
            ).to(self.device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 损失函数
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
        # MONAI 1.4+：FocalLoss 使用 use_softmax 参数
        self.focal_loss = FocalLoss(to_onehot_y=True, use_softmax=True)
        
        # 推理器
        roi_size = (96, 96, 96) if spatial_dims == 3 else (256, 256)
        self.inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=4,
            overlap=0.5,
        )
        
        # 加载预训练模型
        if model_path:
            self.load_model(model_path)
        
        logger.info(
            f"图像分割器已初始化，模型类型: {model_type}, "
            f"空间维度: {spatial_dims}"
        )
    
    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像
            y: 真实标签（训练时）
        
        Returns:
            包含预测结果和损失的字典
        """
        logits = self.model(x)
        output = {"logits": logits}
        
        if y is not None:
            # 计算损失
            dice_loss = self.dice_loss(logits, y)
            focal_loss = self.focal_loss(logits, y)
            output["loss"] = dice_loss + focal_loss
        
        return output
    
    def segment(
        self, image: torch.Tensor, return_probabilities: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        执行分割
        
        Args:
            image: 输入图像
            return_probabilities: 是否返回概率图
        
        Returns:
            分割结果字典
        """
        self.eval()
        with torch.no_grad():
            # 确保输入在正确的设备上
            if isinstance(image, torch.Tensor):
                image = image.to(self.device)
            
            # 添加批次维度
            if len(image.shape) == self.spatial_dims + 1:
                # 已经有通道维度
                if image.shape[0] != 1:
                    image = image.unsqueeze(0)
            elif len(image.shape) == self.spatial_dims:
                image = image.unsqueeze(0).unsqueeze(0)
            
            # 推理
            if self.training:
                logits = self.model(image)
            else:
                logits = self.inferer(image, self.model)
            
            # 获取预测结果
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            output = {"segmentation": predictions}
            
            if return_probabilities:
                output["probabilities"] = probs
            
            return output
    
    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            predictions: 预测结果
            targets: 真实标签
        
        Returns:
            损失字典
        """
        dice_loss = self.dice_loss(predictions, targets)
        focal_loss = self.focal_loss(predictions, targets)
        
        return {
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
            "total_loss": dice_loss + focal_loss,
        }
    
    def load_model(self, model_path: str) -> None:
        """加载预训练模型"""
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
        """保存模型"""
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"模型已保存到 {model_path}")

