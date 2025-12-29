#!/bin/bash
# 运行测试脚本

echo "运行单元测试..."
pytest tests/unit -v

echo "运行集成测试..."
pytest tests/integration -v

echo "生成覆盖率报告..."
pytest --cov=src --cov-report=html --cov-report=term

