"""FastAPI 主应用"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config.settings import settings
from src.api.routes import upload, analyze, diagnosis, treatment, risk, demo
from src.utils.logger import logger

# 创建 FastAPI 应用
app = FastAPI(
    title="多模态医疗诊断辅助系统 API",
    description="提供多模态医疗数据分析和诊断辅助服务",
    version="0.1.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务（用于前端界面）
try:
    import os
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"静态文件目录已挂载: {static_dir}")
except Exception as e:
    logger.warning(f"静态文件目录挂载失败: {e}")

# 注册路由
app.include_router(upload.router, prefix="/api/v1/upload", tags=["数据上传"])
app.include_router(analyze.router, prefix="/api/v1/analyze", tags=["数据分析"])
app.include_router(diagnosis.router, prefix="/api/v1/diagnosis", tags=["诊断"])
app.include_router(treatment.router, prefix="/api/v1/treatment", tags=["治疗方案"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["风险预测"])
app.include_router(demo.router, prefix="/api/v1/demo", tags=["演示/演示数据"])


@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("多模态医疗诊断辅助系统 API 启动")
    logger.info(f"API 地址: http://{settings.API_HOST}:{settings.API_PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("多模态医疗诊断辅助系统 API 关闭")


@app.get("/")
async def root():
    """根路径 - 重定向到前端界面"""
    from fastapi.responses import FileResponse
    import os
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "message": "多模态医疗诊断辅助系统 API",
        "version": "0.1.0",
        "docs": "/docs",
        "ui": "/static/index.html",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

