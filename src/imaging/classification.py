"""图像分类模块"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet, ResNet

from src.config.settings import settings
from src.utils.logger import logger


class ImageClassifier(nn.Module):
    """
    医学图像分类器
    
    支持多种影像模态的分类任务
    """
    
    def __init__(
        self,
        model_type: str = "densenet",
        spatial_dims: int = 2,
        in_channels: int = 1,
        num_classes: int = 2,
        model_path: Optional[str] = None,
    ):
        """
        初始化图像分类器
        
        Args:
            model_type: 模型类型（'densenet' 或 'resnet'）
            spatial_dims: 空间维度（2 或 3）
            in_channels: 输入通道数
            num_classes: 分类类别数
            model_path: 预训练模型路径
        """
        super().__init__()
        
        self.model_type = model_type
        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        self.device = torch.device(settings.DEVICE)
        
        # 构建模型
        if model_type == "densenet":
            self.backbone = DenseNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_classes,
            ).to(self.device)
        elif model_type == "resnet":
            self.backbone = ResNet(
                block="bottleneck",
                layers=(3, 4, 6, 3),
                block_inplanes=(64, 128, 256, 512),
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=num_classes,
            ).to(self.device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载预训练模型
        if model_path:
            self.load_model(model_path)
        
        logger.info(
            f"图像分类器已初始化，模型类型: {model_type}, "
            f"类别数: {num_classes}"
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
        logits = self.backbone(x)
        probs = F.softmax(logits, dim=1)
        
        output = {
            "logits": logits,
            "probabilities": probs,
            "predictions": torch.argmax(probs, dim=1),
        }
        
        if y is not None:
            loss = F.cross_entropy(logits, y)
            output["loss"] = loss
        
        return output
    
    def classify(
        self, image: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        执行分类
        
        Args:
            image: 输入图像
            return_features: 是否返回特征
        
        Returns:
            分类结果字典
        """
        self.eval()
        with torch.no_grad():
            # 确保输入在正确的设备上
            if isinstance(image, torch.Tensor):
                image = image.to(self.device)
            
            # 添加批次维度
            if len(image.shape) == self.spatial_dims:
                image = image.unsqueeze(0).unsqueeze(0)
            elif len(image.shape) == self.spatial_dims + 1:
                if image.shape[0] != 1:
                    image = image.unsqueeze(0)
            
            # 前向传播
            logits = self.backbone(image)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            output = {
                "predictions": predictions,
                "probabilities": probs,
                "class_probs": probs[0].cpu().numpy(),
            }
            
            if return_features:
                # 提取特征（需要修改模型以支持特征提取）
                output["features"] = logits
            
            return output
    
    def get_top_k_classes(
        self, image: torch.Tensor, k: int = 5
    ) -> List[Dict[str, float]]:
        """
        获取 top-k 分类结果
        
        Args:
            image: 输入图像
            k: 返回前 k 个结果
        
        Returns:
            top-k 分类结果列表
        """
        result = self.classify(image)
        probs = result["class_probs"]
        
        # 获取 top-k
        top_k_indices = probs.argsort()[-k:][::-1]
        top_k_probs = probs[top_k_indices]
        
        return [
            {"class": int(idx), "probability": float(prob)}
            for idx, prob in zip(top_k_indices, top_k_probs)
        ]
    
    def load_model(self, model_path: str) -> None:
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.backbone.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.backbone.load_state_dict(checkpoint)
            logger.info(f"模型已从 {model_path} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
    
    def save_model(self, model_path: str) -> None:
        """保存模型"""
        torch.save(self.backbone.state_dict(), model_path)
        logger.info(f"模型已保存到 {model_path}")

