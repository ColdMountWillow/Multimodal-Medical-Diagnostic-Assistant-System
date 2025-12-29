"""缓存工具模块"""
from typing import Optional, Any
import pickle
import hashlib
import json
from functools import wraps
import redis
from datetime import timedelta

from src.config.settings import settings
from src.utils.logger import logger


class CacheManager:
    """
    缓存管理器
    
    提供 Redis 缓存功能
    """
    
    def __init__(self):
        """初始化缓存管理器"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=False,  # 使用二进制模式以支持 pickle
            )
            self.redis_client.ping()
            self.enabled = True
            logger.info("Redis 缓存已连接")
        except Exception as e:
            logger.warning(f"Redis 连接失败: {e}，将禁用缓存")
            self.redis_client = None
            self.enabled = False
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
        
        Returns:
            缓存值或 None
        """
        if not self.enabled:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒）
        
        Returns:
            是否成功
        """
        if not self.enabled:
            return False
        
        try:
            serialized_value = pickle.dumps(value)
            if expire:
                self.redis_client.setex(key, expire, serialized_value)
            else:
                self.redis_client.set(key, serialized_value)
            return True
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
        
        Returns:
            是否成功
        """
        if not self.enabled:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        按模式清除缓存
        
        Args:
            pattern: 键模式（如 "cache:*"）
        
        Returns:
            清除的键数量
        """
        if not self.enabled:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")
            return 0


# 全局缓存管理器实例
cache_manager = CacheManager()


def cached(
    expire: int = 3600,
    key_prefix: str = "cache",
    key_func: Optional[callable] = None,
):
    """
    缓存装饰器
    
    Args:
        expire: 过期时间（秒）
        key_prefix: 键前缀
        key_func: 自定义键生成函数
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认键生成：基于函数名和参数
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": json.dumps(kwargs, sort_keys=True, default=str),
                }
                key_str = json.dumps(key_data, sort_keys=True)
                key_hash = hashlib.md5(key_str.encode()).hexdigest()
                cache_key = f"{key_prefix}:{func.__name__}:{key_hash}"
            
            # 尝试从缓存获取
            cached_value = cache_manager.get(cache_key)
            if cached_value is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_value
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            cache_manager.set(cache_key, result, expire=expire)
            
            return result
        
        return wrapper
    return decorator

