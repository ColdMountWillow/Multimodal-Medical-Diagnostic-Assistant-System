# 多模态医疗诊断辅助系统 - Cursor AI 开发提示词

## 项目概述

### 项目目标
构建一个多模态医疗诊断辅助系统，能够融合多种医疗数据类型（医学影像、实验室检查、病历文本、生理信号等），提供准确、全面的诊断建议和治疗方案推荐。

### 核心价值
- **多模态数据融合**：整合 CT、MRI、X光片、病理切片、心电图、血液检查等多种医疗数据
- **智能诊断辅助**：模拟医生诊断思维，提供诊断建议和依据
- **个性化治疗**：基于患者特征推荐个性化治疗方案
- **知识驱动**：集成医学教科书、临床指南、研究文献等医学知识

## 技术架构

### 技术栈
- **深度学习框架**：PyTorch
- **医学影像处理**：MONAI (Medical Open Network for AI)
- **多模态学习**：hi-ml (Health Intelligence Machine Learning)
- **数据处理**：NumPy, Pandas, DICOM 处理库
- **文本处理**：Transformers, BERT/BioBERT
- **时序分析**：PyTorch Lightning, TensorFlow
- **API 框架**：FastAPI
- **数据库**：PostgreSQL (结构化数据), MongoDB (非结构化数据)
- **缓存**：Redis

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     前端展示层                                │
│              (Web UI / Mobile App)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     API 服务层                               │
│                    (FastAPI RESTful)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   业务逻辑层                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ 多模态融合   │ │ 诊断推理引擎  │ │ 治疗方案推荐 │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   数据处理层                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 影像分析 │ │ 文本挖掘 │ │ 时序分析 │ │ 知识库   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   数据存储层                                  │
│         (PostgreSQL / MongoDB / 文件存储)                    │
└─────────────────────────────────────────────────────────────┘
```

## 核心功能模块

### 1. 多模态数据融合模块

**功能描述**：
整合 CT、MRI、X光片、病理切片、心电图、血液检查等多种医疗数据，实现统一的数据表示和特征提取。

**技术要求**：
- 使用 MONAI 处理医学影像数据（DICOM 格式）
- 使用 hi-ml-multimodal 进行多模态特征融合
- 实现数据对齐和标准化
- 支持异步数据加载和预处理

**代码结构**：
```
src/
├── multimodal/
│   ├── __init__.py
│   ├── data_loader.py          # 多模态数据加载器
│   ├── fusion.py               # 多模态融合模型
│   ├── preprocessing.py        # 数据预处理
│   └── alignment.py            # 数据对齐
```

**实现要点**：
- 使用 MONAI 的 `DataLoader` 和 `Dataset` 类
- 实现自定义的 `MultimodalDataset` 类
- 使用注意力机制进行特征融合
- 支持不同模态数据的缺失处理

### 2. 医学影像分析模块

**功能描述**：
实现病灶检测、分割、分类，支持多种影像模态（CT、MRI、X光片、病理切片）的分析。

**技术要求**：
- 使用 MONAI 的预训练模型和网络架构
- 支持 2D 和 3D 影像处理
- 实现数据增强和变换
- 支持批量推理和实时分析

**代码结构**：
```
src/
├── imaging/
│   ├── __init__.py
│   ├── detection.py            # 病灶检测
│   ├── segmentation.py         # 图像分割
│   ├── classification.py       # 图像分类
│   ├── models/                 # 模型定义
│   │   ├── unet.py
│   │   ├── resnet.py
│   │   └── transformer.py
│   └── transforms.py           # 数据变换
```

**实现要点**：
- 使用 MONAI 的 `UNet`, `DenseNet`, `ResNet` 等网络
- 实现自定义的损失函数（Dice Loss, Focal Loss）
- 使用 MONAI Transforms 进行数据增强
- 支持 GPU 加速和分布式训练

### 3. 病历文本挖掘模块

**功能描述**：
从电子病历中提取症状、诊断、治疗等关键医疗信息，进行命名实体识别和关系抽取。

**技术要求**：
- 使用 BioBERT 或中文医疗 BERT 模型
- 实现命名实体识别（NER）
- 实现关系抽取和知识图谱构建
- 支持结构化病历解析

**代码结构**：
```
src/
├── nlp/
│   ├── __init__.py
│   ├── ner.py                  # 命名实体识别
│   ├── relation_extraction.py  # 关系抽取
│   ├── text_classification.py  # 文本分类
│   ├── models/
│   │   └── bert_model.py
│   └── preprocessing.py        # 文本预处理
```

**实现要点**：
- 使用 Transformers 库加载预训练模型
- 实现医疗领域特定的 tokenizer
- 使用 CRF 或 BiLSTM-CRF 进行序列标注
- 实现医疗知识图谱构建

### 4. 时序数据分析模块

**功能描述**：
分析生理信号、监护数据等时间序列医疗数据，进行趋势预测和异常检测。

**技术要求**：
- 使用 LSTM、GRU、Transformer 等时序模型
- 实现时间序列预处理和特征提取
- 支持多变量时间序列分析
- 实现异常检测算法

**代码结构**：
```
src/
├── timeseries/
│   ├── __init__.py
│   ├── preprocessing.py        # 时序数据预处理
│   ├── models/
│   │   ├── lstm.py
│   │   ├── transformer.py
│   │   └── anomaly_detection.py
│   └── analysis.py             # 时序分析
```

**实现要点**：
- 使用 PyTorch 实现 LSTM/GRU 模型
- 实现滑动窗口和特征工程
- 使用注意力机制处理长序列
- 实现异常检测和预警机制

### 5. 疾病风险预测模块

**功能描述**：
基于多模态数据预测疾病发展趋势和并发症风险，提供风险评估和预警。

**技术要求**：
- 融合多模态特征进行风险预测
- 使用集成学习方法提高预测准确性
- 实现风险评分和可视化
- 支持多疾病风险预测

**代码结构**：
```
src/
├── prediction/
│   ├── __init__.py
│   ├── risk_prediction.py      # 风险预测模型
│   ├── feature_engineering.py  # 特征工程
│   └── evaluation.py           # 模型评估
```

**实现要点**：
- 使用 XGBoost、LightGBM 进行风险预测
- 实现 SHAP 值进行特征重要性分析
- 使用生存分析模型（Cox 回归）
- 实现风险分层和预警阈值设置

### 6. 诊断推理引擎模块

**功能描述**：
模拟医生的诊断思维过程，提供诊断建议和依据，支持可解释性诊断。

**技术要求**：
- 实现基于规则的推理引擎
- 集成机器学习模型进行诊断辅助
- 提供诊断路径和依据解释
- 支持多疾病诊断和鉴别诊断

**代码结构**：
```
src/
├── reasoning/
│   ├── __init__.py
│   ├── rule_engine.py          # 规则推理引擎
│   ├── ml_engine.py            # 机器学习推理
│   ├── explanation.py          # 可解释性模块
│   └── diagnosis_path.py       # 诊断路径生成
```

**实现要点**：
- 实现知识图谱查询和推理
- 使用图神经网络进行诊断推理
- 实现诊断置信度计算
- 提供诊断依据的可视化展示

### 7. 治疗方案推荐模块

**功能描述**：
根据诊断结果和患者特征推荐个性化的治疗方案，考虑药物相互作用和禁忌症。

**技术要求**：
- 基于诊断结果和患者信息推荐治疗方案
- 考虑药物相互作用和禁忌症
- 支持治疗方案优化和调整
- 集成临床指南和循证医学证据

**代码结构**：
```
src/
├── treatment/
│   ├── __init__.py
│   ├── recommendation.py       # 治疗方案推荐
│   ├── drug_interaction.py     # 药物相互作用检查
│   ├── guideline_engine.py     # 临床指南引擎
│   └── optimization.py         # 方案优化
```

**实现要点**：
- 使用推荐系统算法（协同过滤、内容推荐）
- 实现药物知识库和相互作用检查
- 集成临床指南数据库
- 实现治疗方案评分和排序

### 8. 医疗知识集成模块

**功能描述**：
集成医学教科书、临床指南、研究文献等医学知识，构建知识库和知识图谱。

**技术要求**：
- 实现知识抽取和结构化存储
- 构建医疗知识图谱
- 实现知识检索和查询接口
- 支持知识更新和版本管理

**代码结构**：
```
src/
├── knowledge/
│   ├── __init__.py
│   ├── extraction.py           # 知识抽取
│   ├── graph.py                # 知识图谱
│   ├── retrieval.py            # 知识检索
│   └── storage.py              # 知识存储
```

**实现要点**：
- 使用 Neo4j 或图数据库存储知识图谱
- 实现知识抽取管道（实体识别、关系抽取）
- 实现语义搜索和相似度计算
- 支持知识验证和更新机制

## 开发规范

### 代码规范
- 遵循 PEP 8 Python 代码规范
- 使用类型提示（Type Hints）
- 编写详细的文档字符串（Docstrings）
- 使用有意义的变量和函数名
- 每个函数不超过 50 行代码

### 文件结构
```
Multimodal-Medical-Diagnostic-Assistant-System/
├── src/                        # 源代码目录
│   ├── multimodal/            # 多模态融合模块
│   ├── imaging/               # 医学影像分析模块
│   ├── nlp/                   # 病历文本挖掘模块
│   ├── timeseries/            # 时序数据分析模块
│   ├── prediction/            # 疾病风险预测模块
│   ├── reasoning/             # 诊断推理引擎模块
│   ├── treatment/             # 治疗方案推荐模块
│   ├── knowledge/             # 医疗知识集成模块
│   ├── api/                   # API 接口层
│   ├── utils/                 # 工具函数
│   └── config/                # 配置文件
├── tests/                      # 测试代码
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   └── models/                # 模型文件
├── docs/                       # 文档目录
├── scripts/                    # 脚本文件
├── requirements.txt            # Python 依赖
├── setup.py                    # 安装脚本
└── README.md                   # 项目说明
```

### 命名约定
- **模块名**：小写字母，单词间用下划线（snake_case）
- **类名**：大驼峰命名（PascalCase）
- **函数名**：小写字母，单词间用下划线（snake_case）
- **常量名**：全大写字母，单词间用下划线（UPPER_SNAKE_CASE）
- **私有变量/函数**：以下划线开头

### 依赖管理

**核心依赖**：
```python
# requirements.txt
torch>=2.0.0
monai>=1.3.0
transformers>=4.30.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
redis>=4.6.0
psycopg2-binary>=2.9.0
pymongo>=4.4.0
```

**MONAI 使用示例**：
```python
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImage, AddChannel, ScaleIntensity
from monai.networks.nets import UNet
```

**hi-ml 使用示例**：
```python
from health_multimodal.image import ImageModel
from health_multimodal.text import TextModel
from health_multimodal.multimodal import MultiModalModel
```

## 数据流程

### 多模态数据处理流程

```
原始数据输入
    │
    ├── 医学影像 (DICOM)
    │   └── MONAI 预处理 → 特征提取
    │
    ├── 病历文本
    │   └── 文本清洗 → BERT 编码 → 特征提取
    │
    ├── 实验室检查数据
    │   └── 数据标准化 → 特征工程
    │
    └── 生理信号 (时间序列)
        └── 时序预处理 → LSTM/Transformer → 特征提取
            │
            └── 多模态特征融合
                │
                └── 诊断推理引擎
                    │
                    ├── 诊断结果
                    ├── 风险评估
                    └── 治疗方案推荐
```

## 模型设计

### 多模态融合模型架构

**建议架构**：
- **编码器层**：各模态独立编码器（CNN for 影像，BERT for 文本，LSTM for 时序）
- **融合层**：注意力机制或 Transformer 融合层
- **解码器层**：任务特定的解码器（分类、回归、生成）

**示例代码结构**：
```python
class MultimodalFusionModel(nn.Module):
    def __init__(self):
        self.image_encoder = ImageEncoder()  # MONAI UNet/ResNet
        self.text_encoder = TextEncoder()    # BERT
        self.timeseries_encoder = TimeseriesEncoder()  # LSTM
        self.fusion_layer = AttentionFusion()
        self.classifier = ClassificationHead()
    
    def forward(self, image, text, timeseries):
        img_features = self.image_encoder(image)
        txt_features = self.text_encoder(text)
        ts_features = self.timeseries_encoder(timeseries)
        fused = self.fusion_layer(img_features, txt_features, ts_features)
        return self.classifier(fused)
```

## API 设计

### RESTful API 接口规范

**基础路径**：`/api/v1`

**主要接口**：

1. **数据上传接口**
   - `POST /api/v1/upload/image` - 上传医学影像
   - `POST /api/v1/upload/text` - 上传病历文本
   - `POST /api/v1/upload/timeseries` - 上传时序数据

2. **分析接口**
   - `POST /api/v1/analyze/multimodal` - 多模态综合分析
   - `POST /api/v1/analyze/image` - 影像分析
   - `POST /api/v1/analyze/text` - 文本分析
   - `POST /api/v1/analyze/timeseries` - 时序分析

3. **诊断接口**
   - `POST /api/v1/diagnosis/predict` - 诊断预测
   - `GET /api/v1/diagnosis/{diagnosis_id}` - 获取诊断结果
   - `GET /api/v1/diagnosis/{diagnosis_id}/explanation` - 获取诊断依据

4. **治疗方案接口**
   - `POST /api/v1/treatment/recommend` - 治疗方案推荐
   - `GET /api/v1/treatment/{treatment_id}` - 获取治疗方案详情

5. **风险预测接口**
   - `POST /api/v1/risk/predict` - 疾病风险预测
   - `GET /api/v1/risk/{patient_id}/history` - 获取风险历史

**API 响应格式**：
```json
{
    "status": "success",
    "data": {
        // 响应数据
    },
    "message": "操作成功",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

## 测试策略

### 单元测试
- 每个模块至少 80% 代码覆盖率
- 使用 pytest 框架
- 测试文件命名：`test_*.py`

### 集成测试
- 测试多模态数据融合流程
- 测试端到端诊断流程
- 测试 API 接口

### 测试示例结构
```
tests/
├── unit/
│   ├── test_multimodal.py
│   ├── test_imaging.py
│   └── test_nlp.py
├── integration/
│   ├── test_diagnosis_flow.py
│   └── test_api.py
└── fixtures/
    └── sample_data.py
```

## 部署指南

### 开发环境
- Python 3.9+
- CUDA 11.8+ (GPU 支持)
- Docker 和 Docker Compose

### 生产环境
- 使用 Docker 容器化部署
- 使用 Kubernetes 进行容器编排
- 配置 GPU 资源分配
- 设置监控和日志系统

### Dockerfile 示例
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 开发工作流

### Git 工作流
- 主分支：`main`（生产环境）
- 开发分支：`develop`（开发环境）
- 功能分支：`feature/模块名-功能描述`
- 修复分支：`fix/问题描述`

### 提交信息规范
- `feat: 添加新功能`
- `fix: 修复问题`
- `docs: 更新文档`
- `test: 添加测试`
- `refactor: 代码重构`

## 注意事项

### 医疗数据安全
- 严格遵守 HIPAA、GDPR 等数据保护法规
- 实现数据加密和访问控制
- 记录数据访问日志
- 定期进行安全审计

### 模型可解释性
- 提供诊断依据和推理路径
- 使用 SHAP、LIME 等可解释性工具
- 可视化特征重要性
- 记录模型决策过程

### 性能优化
- 使用 GPU 加速推理
- 实现模型量化和剪枝
- 使用缓存机制减少重复计算
- 实现异步处理和批处理

### 错误处理
- 实现完善的异常处理机制
- 提供友好的错误提示
- 记录错误日志用于调试
- 实现降级策略

## 参考资源

### 开源项目
- **MONAI**: https://github.com/Project-MONAI/MONAI
- **hi-ml**: https://github.com/microsoft/hi-ml

### 文档资源
- MONAI 文档: https://docs.monai.io/
- PyTorch 文档: https://pytorch.org/docs/
- FastAPI 文档: https://fastapi.tiangolo.com/

### 数据集
- MIMIC-III (需要申请)
- ChestX-ray14
- ISIC (皮肤病变)
- 其他公开医疗数据集

## 开发优先级

### Phase 1: 基础模块开发（1-2个月）
1. 多模态数据融合模块
2. 医学影像分析模块
3. 基础 API 接口

### Phase 2: 核心功能开发（2-3个月）
4. 病历文本挖掘模块
5. 时序数据分析模块
6. 诊断推理引擎模块

### Phase 3: 高级功能开发（2-3个月）
7. 疾病风险预测模块
8. 治疗方案推荐模块
9. 医疗知识集成模块

### Phase 4: 优化和部署（1-2个月）
10. 性能优化
11. 系统测试
12. 生产环境部署

---

**提示**：在开发过程中，始终遵循医疗 AI 的伦理和安全原则，确保系统的可靠性和可解释性。定期与医疗专家沟通，验证系统的准确性和实用性。

