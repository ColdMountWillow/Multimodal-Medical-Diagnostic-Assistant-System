"""系统配置"""
import os
from pathlib import Path
from typing import Optional


class Settings:
    """应用配置"""
    
    def __init__(self):
        """初始化配置"""
        # 项目根目录
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        
        # API 配置
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"
        
        # 数据库配置
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "medical_db")
        
        self.MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
        self.MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
        self.MONGODB_DB = os.getenv("MONGODB_DB", "medical_db")
        
        # Redis 配置
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        
        # 模型配置
        self.MODEL_DIR = self.PROJECT_ROOT / "data" / "models"
        self.DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
        
        # 数据目录
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        
        # 日志配置
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_DIR = self.PROJECT_ROOT / "logs"
        
        # 安全配置
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
        )


# 全局配置实例
settings = Settings()
