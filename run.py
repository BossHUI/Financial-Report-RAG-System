#-*- coding: utf-8 -*-
import argparse
from src.main import main
from src.config import FINANCIAL_REPORTS_DIR, EMBEDDING_FILE

def parse_args():
    parser = argparse.ArgumentParser(description="运行RAG系统")
    parser.add_argument('--mode', choices=['train', 'interactive', 'api', 'test_model'],
                      default='interactive', help="运行模式：train(处理文档)、interactive(交互式问答)、api(API服务)或test_model(测试本地模型)")
    parser.add_argument('--folder_path', default=FINANCIAL_REPORTS_DIR,
                      help="PDF文件目录路径")
    parser.add_argument('--embedding_file', default=EMBEDDING_FILE,
                      help="嵌入向量存储路径")
    parser.add_argument('--port', type=int, default=8000,
                      help="API服务端口号")
    parser.add_argument('--use_api', action='store_true',
                      help="强制使用OpenAI API，即使本地模型存在")
    parser.add_argument('--use_local', action='store_true',
                      help="强制使用本地模型，如果不可用则报错")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)