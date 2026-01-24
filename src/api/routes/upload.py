"""数据上传接口"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import uuid
import os
from datetime import datetime
from pathlib import Path

from src.config.settings import settings
from src.utils.logger import logger

router = APIRouter()

# 创建上传文件存储目录
UPLOAD_DIR = settings.DATA_DIR / "raw" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    """
    上传医学影像
    
    Args:
        file: 图像文件
    
    Returns:
        上传结果
    """
    try:
        # 生成唯一 ID
        file_id = str(uuid.uuid4())
        
        # 获取文件扩展名
        file_ext = Path(file.filename).suffix if file.filename else ""
        saved_filename = f"{file_id}{file_ext}"
        saved_path = UPLOAD_DIR / saved_filename
        
        # 保存文件
        file_content = await file.read()
        with open(saved_path, "wb") as f:
            f.write(file_content)
        
        file_size = len(file_content)
        
        logger.info(f"收到图像上传请求: {file.filename}, ID: {file_id}, 大小: {file_size} bytes")
        
        return {
            "status": "success",
            "data": {
                "file_id": file_id,
                "filename": file.filename,
                "saved_filename": saved_filename,
                "saved_path": str(saved_path),
                "file_size": file_size,
                "content_type": file.content_type,
                "upload_time": datetime.now().isoformat(),
            },
            "message": "图像上传成功",
        }
    except Exception as e:
        logger.error(f"图像上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/text")
async def upload_text(text: str):
    """
    上传病历文本
    
    Args:
        text: 病历文本内容
    
    Returns:
        上传结果
    """
    try:
        text_id = str(uuid.uuid4())
        
        # 保存文本到文件
        text_file = UPLOAD_DIR / f"{text_id}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        logger.info(f"收到文本上传请求, ID: {text_id}, 长度: {len(text)} 字符")
        
        return {
            "status": "success",
            "data": {
                "text_id": text_id,
                "length": len(text),
                "saved_path": str(text_file),
                "upload_time": datetime.now().isoformat(),
            },
            "message": "文本上传成功",
        }
    except Exception as e:
        logger.error(f"文本上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/timeseries")
async def upload_timeseries(data: dict):
    """
    上传时序数据
    
    Args:
        data: 时序数据（JSON 格式）
    
    Returns:
        上传结果
    """
    try:
        timeseries_id = str(uuid.uuid4())
        
        logger.info(f"收到时序数据上传请求, ID: {timeseries_id}")
        
        return {
            "status": "success",
            "data": {
                "timeseries_id": timeseries_id,
                "data_points": len(data.get("values", [])),
                "upload_time": datetime.now().isoformat(),
            },
            "message": "时序数据上传成功",
        }
    except Exception as e:
        logger.error(f"时序数据上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

