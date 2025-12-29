@echo off
REM 构建 Docker 镜像脚本

echo 构建 Docker 镜像...
docker build -t multimodal-medical-api:latest .

echo 镜像构建完成！
echo 运行容器: docker-compose up -d

