"""数据准备脚本"""
"""
数据准备脚本

使用方法：
    python scripts/prepare_data.py --dataset chestxray --output_dir data/processed
    python scripts/prepare_data.py --dataset mimic --output_dir data/processed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger
from src.config.settings import settings


def prepare_chestxray_data(input_dir: str, output_dir: str):
    """准备ChestX-ray14数据集"""
    logger.info("准备ChestX-ray14数据集...")
    # TODO: 实现数据准备逻辑
    # 1. 下载数据集（如果需要）
    # 2. 数据清洗
    # 3. 数据分割（训练/验证/测试）
    # 4. 数据格式转换
    logger.info("ChestX-ray14数据准备完成")


def prepare_mimic_data(input_dir: str, output_dir: str):
    """准备MIMIC-III数据集"""
    logger.info("准备MIMIC-III数据集...")
    # TODO: 实现数据准备逻辑
    # 注意：MIMIC-III需要申请权限
    logger.info("MIMIC-III数据准备完成")


def prepare_custom_data(input_dir: str, output_dir: str):
    """准备自定义数据集"""
    logger.info("准备自定义数据集...")
    # TODO: 实现数据准备逻辑
    logger.info("自定义数据准备完成")


def main():
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["chestxray", "mimic", "custom"],
        help="数据集类型",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="输入数据目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(settings.PROCESSED_DATA_DIR),
        help="输出数据目录",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        help="数据分割比例（训练/验证/测试）",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("数据准备开始")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 50)
    
    try:
        if args.dataset == "chestxray":
            prepare_chestxray_data(args.input_dir, args.output_dir)
        
        elif args.dataset == "mimic":
            prepare_mimic_data(args.input_dir, args.output_dir)
        
        elif args.dataset == "custom":
            if not args.input_dir:
                raise ValueError("自定义数据集需要指定--input_dir")
            prepare_custom_data(args.input_dir, args.output_dir)
        
        logger.info("=" * 50)
        logger.info("数据准备完成")
        logger.info("=" * 50)
    
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        raise


if __name__ == "__main__":
    main()

