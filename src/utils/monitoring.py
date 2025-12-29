"""监控模块"""
from typing import Dict, Optional
import time
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
import psutil
import GPUtil

from src.utils.logger import logger


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    active_requests: int = 0
    total_requests: int = 0


class SystemMonitor:
    """
    系统监控器
    
    监控系统资源使用情况
    """
    
    def __init__(self):
        """初始化监控器"""
        self.process = psutil.Process()
        self.gpu_available = self._check_gpu_available()
        logger.info("系统监控器已初始化")
    
    def _check_gpu_available(self) -> bool:
        """检查 GPU 是否可用"""
        try:
            GPUtil.getGPUs()
            return True
        except:
            return False
    
    def get_metrics(self) -> PerformanceMetrics:
        """
        获取当前性能指标
        
        Returns:
            性能指标对象
        """
        # CPU 和内存
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        metrics = PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_info.rss / 1024 / 1024,
        )
        
        # GPU（如果可用）
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.gpu_percent = gpu.load * 100
                    metrics.gpu_memory_percent = (
                        gpu.memoryUsed / gpu.memoryTotal * 100
                    )
            except Exception as e:
                logger.warning(f"获取 GPU 指标失败: {e}")
        
        return metrics
    
    def get_system_info(self) -> Dict[str, any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "disk_total_gb": psutil.disk_usage("/").total / 1024 / 1024 / 1024,
            "gpu_available": self.gpu_available,
        }


class RequestMonitor:
    """
    请求监控器
    
    监控 API 请求的性能
    """
    
    def __init__(self):
        """初始化请求监控器"""
        self.request_count = 0
        self.request_times = []
        self.error_count = 0
        logger.info("请求监控器已初始化")
    
    def record_request(self, duration: float, success: bool = True):
        """
        记录请求
        
        Args:
            duration: 请求耗时（秒）
            success: 是否成功
        """
        self.request_count += 1
        self.request_times.append(duration)
        
        if not success:
            self.error_count += 1
        
        # 只保留最近 1000 个请求的时间
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
    
    def get_stats(self) -> Dict[str, any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        if not self.request_times:
            return {
                "total_requests": 0,
                "error_count": 0,
                "avg_duration": 0.0,
                "p95_duration": 0.0,
                "p99_duration": 0.0,
            }
        
        sorted_times = sorted(self.request_times)
        
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0.0,
            "avg_duration": sum(self.request_times) / len(self.request_times),
            "min_duration": min(self.request_times),
            "max_duration": max(self.request_times),
            "p95_duration": sorted_times[int(len(sorted_times) * 0.95)],
            "p99_duration": sorted_times[int(len(sorted_times) * 0.99)],
        }


# 全局监控器实例
system_monitor = SystemMonitor()
request_monitor = RequestMonitor()


def monitor_performance(func):
    """
    性能监控装饰器
    
    Args:
        func: 被装饰的函数
    
    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
            raise
        finally:
            duration = time.time() - start_time
            request_monitor.record_request(duration, success)
            
            if duration > 1.0:  # 记录慢请求
                logger.warning(
                    f"慢请求警告: {func.__name__} 耗时 {duration:.2f} 秒"
                )
    
    return wrapper

