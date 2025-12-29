"""构建知识库脚本"""
"""
构建医疗知识库脚本

使用方法：
    python scripts/build_knowledge_base.py --source guidelines --output data/knowledge
    python scripts/build_knowledge_base.py --source text --input_dir data/texts
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.extraction import KnowledgeExtractor
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.storage import KnowledgeStorage
from src.utils.logger import logger
from src.config.settings import settings


def build_from_guidelines(output_path: str):
    """从临床指南构建知识库"""
    logger.info("从临床指南构建知识库...")
    
    extractor = KnowledgeExtractor()
    graph = KnowledgeGraph()
    storage = KnowledgeStorage(Path(output_path))
    
    # TODO: 加载指南数据
    # guidelines = load_guidelines()
    # for guideline in guidelines:
    #     entities, relations = extractor.extract_from_text(
    #         guideline.content, source=guideline.source
    #     )
    #     for entity in entities:
    #         graph.add_entity(entity)
    #     for relation in relations:
    #         graph.add_relation(relation)
    
    # 保存知识图谱
    storage.save_graph(graph)
    logger.info("知识库构建完成")


def build_from_texts(input_dir: str, output_path: str):
    """从文本文件构建知识库"""
    logger.info("从文本文件构建知识库...")
    
    extractor = KnowledgeExtractor()
    graph = KnowledgeGraph()
    storage = KnowledgeStorage(Path(output_path))
    
    # TODO: 遍历文本文件并提取知识
    # text_files = Path(input_dir).glob("*.txt")
    # for text_file in text_files:
    #     with open(text_file, 'r', encoding='utf-8') as f:
    #         text = f.read()
    #         entities, relations = extractor.extract_from_text(
    #             text, source=str(text_file)
    #         )
    #         for entity in entities:
    #             graph.add_entity(entity)
    #         for relation in relations:
    #             graph.add_relation(relation)
    
    # 保存知识图谱
    storage.save_graph(graph)
    logger.info("知识库构建完成")


def build_from_structured_data(input_file: str, output_path: str):
    """从结构化数据构建知识库"""
    logger.info("从结构化数据构建知识库...")
    
    import json
    
    extractor = KnowledgeExtractor()
    graph = KnowledgeGraph()
    storage = KnowledgeStorage(Path(output_path))
    
    # TODO: 加载结构化数据
    # with open(input_file, 'r', encoding='utf-8') as f:
    #     structured_data = json.load(f)
    #     entities, relations = extractor.extract_structured_knowledge(
    #         structured_data
    #     )
    #     for entity in entities:
    #         graph.add_entity(entity)
    #     for relation in relations:
    #         graph.add_relation(relation)
    
    # 保存知识图谱
    storage.save_graph(graph)
    logger.info("知识库构建完成")


def main():
    parser = argparse.ArgumentParser(description="构建医疗知识库")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["guidelines", "text", "structured"],
        help="知识源类型",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="输入目录（用于text源）",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="输入文件（用于structured源）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(settings.DATA_DIR / "knowledge"),
        help="输出目录",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("知识库构建开始")
    logger.info(f"知识源: {args.source}")
    logger.info(f"输出目录: {args.output}")
    logger.info("=" * 50)
    
    try:
        if args.source == "guidelines":
            build_from_guidelines(args.output)
        
        elif args.source == "text":
            if not args.input_dir:
                raise ValueError("text源需要指定--input_dir")
            build_from_texts(args.input_dir, args.output)
        
        elif args.source == "structured":
            if not args.input_file:
                raise ValueError("structured源需要指定--input_file")
            build_from_structured_data(args.input_file, args.output)
        
        logger.info("=" * 50)
        logger.info("知识库构建完成")
        logger.info("=" * 50)
    
    except Exception as e:
        logger.error(f"知识库构建失败: {e}")
        raise


if __name__ == "__main__":
    main()

