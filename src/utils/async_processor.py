"""异步处理模块"""
from typing import List, Callable, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

from src.utils.logger import logger


class AsyncProcessor:
    """
    异步处理器
    
    提供异步批处理功能
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
    ):
        """
        初始化异步处理器
        
        Args:
            max_workers: 最大工作线程/进程数
            use_processes: 是否使用进程池（用于 CPU 密集型任务）
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(
            f"异步处理器已初始化，工作线程数: {max_workers}, "
            f"使用进程: {use_processes}"
        )
    
    async def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: int = 10,
    ) -> List[Any]:
        """
        批量异步处理
        
        Args:
            items: 待处理项列表
            process_func: 处理函数
            batch_size: 批次大小
        
        Returns:
            处理结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            
            # 创建任务
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(self.executor, process_func, item)
                for item in batch
            ]
            
            # 等待批次完成
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results
    
    def process_batch_sync(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: int = 10,
    ) -> List[Any]:
        """
        同步批量处理
        
        Args:
            items: 待处理项列表
            process_func: 处理函数
            batch_size: 批次大小
        
        Returns:
            处理结果列表
        """
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batch_results = [
                process_func(item) for item in batch
            ]
            results.extend(batch_results)
        
        return results
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        logger.info("异步处理器已关闭")


class BatchProcessor:
    """
    批处理器
    
    提供高效的批处理功能
    """
    
    def __init__(self, batch_size: int = 32):
        """
        初始化批处理器
        
        Args:
            batch_size: 批次大小
        """
        self.batch_size = batch_size
        logger.info(f"批处理器已初始化，批次大小: {batch_size}")
    
    def process_batches(
        self,
        data: List[Any],
        process_func: Callable,
    ) -> List[Any]:
        """
        分批处理数据
        
        Args:
            data: 数据列表
            process_func: 处理函数（接受批次数据）
        
        Returns:
            处理结果列表
        """
        results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]
            batch_result = process_func(batch)
            
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results

