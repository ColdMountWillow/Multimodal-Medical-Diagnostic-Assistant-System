"""启动 API 服务脚本"""
import uvicorn
from src.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
    )

