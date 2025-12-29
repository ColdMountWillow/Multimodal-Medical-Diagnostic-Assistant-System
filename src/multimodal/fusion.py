"""多模态融合模型"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import logger


class AttentionFusion(nn.Module):
    """
    基于注意力机制的多模态融合层
    """
    
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int = 256):
        """
        初始化注意力融合层
        
        Args:
            feature_dims: 各模态特征维度字典
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # 为每个模态创建投影层
        self.projections = nn.ModuleDict()
        for modality, dim in feature_dims.items():
            self.projections[modality] = nn.Linear(dim, hidden_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self, features_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features_dict: 各模态特征字典
        
        Returns:
            融合后的特征
        """
        # 投影到统一维度
        projected_features = []
        for modality, features in features_dict.items():
            if modality in self.projections:
                proj_features = self.projections[modality](features)
                # 添加模态维度
                if len(proj_features.shape) == 2:
                    proj_features = proj_features.unsqueeze(1)
                projected_features.append(proj_features)
        
        if not projected_features:
            raise ValueError("没有可用的模态特征")
        
        # 拼接所有模态特征
        stacked_features = torch.cat(projected_features, dim=1)
        
        # 自注意力
        attended_features, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # 平均池化
        fused_features = torch.mean(attended_features, dim=1)
        
        # 融合层
        output = self.fusion_layer(fused_features)
        
        return output


class MultimodalFusionModel(nn.Module):
    """
    多模态融合模型
    
    整合不同模态的特征，使用注意力机制进行融合
    """
    
    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        timeseries_encoder: Optional[nn.Module] = None,
        fusion_hidden_dim: int = 256,
        num_classes: Optional[int] = None,
    ):
        """
        初始化多模态融合模型
        
        Args:
            image_encoder: 图像编码器
            text_encoder: 文本编码器
            timeseries_encoder: 时间序列编码器
            fusion_hidden_dim: 融合层隐藏维度
            num_classes: 分类类别数（如果为 None，则只返回特征）
        """
        super().__init__()
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.timeseries_encoder = timeseries_encoder
        
        # 确定特征维度
        feature_dims = {}
        if image_encoder:
            # 假设图像编码器输出维度为 512
            feature_dims["image"] = 512
        if text_encoder:
            # 假设文本编码器输出维度为 768（BERT）
            feature_dims["text"] = 768
        if timeseries_encoder:
            # 假设时间序列编码器输出维度为 256
            feature_dims["timeseries"] = 256
        
        # 融合层
        if feature_dims:
            self.fusion_layer = AttentionFusion(
                feature_dims, hidden_dim=fusion_hidden_dim
            )
        else:
            self.fusion_layer = None
        
        # 分类头（可选）
        self.classifier = None
        if num_classes:
            self.classifier = nn.Linear(fusion_hidden_dim, num_classes)
        
        logger.info("多模态融合模型已初始化")
    
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        timeseries: Optional[torch.Tensor] = None,
        lab_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            image: 图像张量
            text: 文本张量或字典
            timeseries: 时间序列张量
            lab_data: 实验室数据张量
        
        Returns:
            包含融合特征和（可选）分类结果的字典
        """
        features_dict = {}
        
        # 提取各模态特征
        if image is not None and self.image_encoder:
            img_features = self.image_encoder(image)
            features_dict["image"] = img_features
        
        if text is not None and self.text_encoder:
            if isinstance(text, dict):
                txt_features = self.text_encoder(**text)
            else:
                txt_features = self.text_encoder(text)
            features_dict["text"] = txt_features
        
        if timeseries is not None and self.timeseries_encoder:
            ts_features = self.timeseries_encoder(timeseries)
            features_dict["timeseries"] = ts_features
        
        # 融合特征
        if self.fusion_layer and features_dict:
            fused_features = self.fusion_layer(features_dict)
        else:
            # 如果没有融合层，直接拼接特征
            if features_dict:
                fused_features = torch.cat(list(features_dict.values()), dim=-1)
            else:
                raise ValueError("没有可用的模态特征")
        
        # 分类（如果指定了类别数）
        output = {"features": fused_features}
        
        if self.classifier:
            logits = self.classifier(fused_features)
            output["logits"] = logits
            output["probs"] = F.softmax(logits, dim=-1)
        
        return output

