
import os
import time
import torch
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    MODELS, 
    DEVICES, 
    DATA_DIR, 
    MODEL_CACHE_DIR,
    BATCH_SIZE,
    MAX_LENGTH,
    FINANCIAL_REPORTS_DIR,
    EMBEDDING_FILE
)
from src.embeddings import EmbeddingProcessor
from src.retrieval import DocumentRetriever
from src.utils import setup_logging, load_env_variables

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """生成配置类"""
    max_new_tokens: int = 256
    num_beams: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    early_stopping: bool = True

class RAGSystem:
    """
    @class RAGSystem
    @description RAG系统主类
    """
    def __init__(self, embedding_file=None):
        """初始化RAG系统"""
        # 加载环境变量
        load_env_variables()
        
        # 初始化组件
        self.embedding_processor = EmbeddingProcessor()
        self.tokenizer, self.model = self.load_generation_model()
        self.document_retriever = DocumentRetriever(embedding_file=embedding_file)
        self.generation_config = GenerationConfig()

    def load_generation_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """加载生成模型"""
        logger.info("加载生成模型...")
        model_path = MODELS["generation"]["path"]
        model_name = MODELS["generation"]["name"]
        
        try:
            # 检查本地模型路径是否存在
            if model_path and os.path.exists(model_path):
                logger.info(f"使用本地生成模型: {model_path}")
                
                # 检查模型文件是否完整
                required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
                missing_files = []
                
                for file in required_files:
                    file_path = os.path.join(model_path, file)
                    if not os.path.exists(file_path):
                        missing_files.append(file)
                
                if missing_files:
                    logger.warning(f"本地模型文件不完整，缺少: {', '.join(missing_files)}")
                    logger.info("将使用OpenAI API作为备选方案")
                    return None, None
                
                # 加载tokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True
                    )
                except Exception as e:
                    logger.error(f"加载tokenizer失败: {str(e)}")
                    return None, None
                
                # 加载模型
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        local_files_only=True
                    ).to(DEVICES)
                except Exception as e:
                    logger.error(f"加载模型失败: {str(e)}")
                    return None, None
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                return tokenizer, model
            
            # 如果本地路径不存在，使用API
            else:
                logger.info(f"未找到本地模型，将使用OpenAI API生成回答: {model_name}")
                return None, None
            
        except Exception as e:
            logger.error(f"加载生成模型失败: {str(e)}")
            logger.info("将使用OpenAI API作为备选方案")
            return None, None

    def generate_answer(
        self, 
        query: str, 
        contexts: List[str]
    ) -> str:
        """
        生成答案
        @param query: 查询问题
        @param contexts: 相关上下文列表
        @return: 生成的答案
        """
        # 组合上下文和问题
        combined_context = "\n\n".join([
            f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)
        ])
        
        # 构建更详细的系统提示
        system_prompt = """你是一位专业的财务分析师，善于分析财报数据。请根据提供的上下文回答问题。
        回答要求：
        1. 只基于提供的上下文信息回答，不要编造信息
        2. 如果上下文不足以回答问题，请说明原因
        3. 如果上下文中有矛盾的信息，请指出并说明
        4. 回答要专业、准确、简洁
        5. 如果涉及数据，请引用具体数字
        6. 如果上下文中有相关数据，即使不是直接回答，也请提供并说明为什么这可能与问题相关
        7. 对于财务相关查询，要积极查找上下文中的数字、百分比、年份等关键数据
        8. 引用数据时，请指明数据来源于哪个上下文
        """
        
        # 构建用户提示
        user_prompt = f"""基于以下上下文信息回答问题：

{combined_context}

问题：{query}

请提供详细的分析和答案。如果上下文中有任何财务数据（如收入、利润、增长率等）与问题相关，请务必提取并引用。"""

        try:
            # 使用OpenAI API生成回答
            from openai import OpenAI
            import os
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("未设置OPENAI_API_KEY，无法使用OpenAI API")
                return "无法生成回答：未设置OpenAI API密钥。请设置环境变量OPENAI_API_KEY。"
            
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content.strip()
                return answer
            except Exception as e:
                logger.error(f"OpenAI API请求失败: {str(e)}")
                return f"无法生成回答：OpenAI API请求失败，错误信息: {str(e)}"
            
        except Exception as e:
            logger.error(f"生成答案时出错: {str(e)}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

    def process_query(
        self, 
        query: str, 
        top_k: int = 3
    ) -> Tuple[str, List[str]]:
        """
        处理查询
        @param query: 查询问题
        @param top_k: 返回的相关文档数量
        @return: (生成的答案, 相关上下文列表)
        """
        try:
            # 检索相关文档
            relevant_docs = self.document_retriever.retrieve(
                query, 
                top_k=top_k
            )
            
            # 生成答案
            answer = self.generate_answer(query, relevant_docs)
            
            return answer, relevant_docs
            
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}")
            return "处理查询时出现错误。", []
    
    def process_financial_reports(self, folder_path, embedding_file_path):
        """
        处理财报文件并生成嵌入向量
        @param folder_path: 财报文件夹路径
        @param embedding_file_path: 嵌入向量存储路径
        @return: 处理的文档数量
        """
        logger.info(f"开始处理财报文件: {folder_path}")
        try:
            documents_with_embeddings = self.embedding_processor.process_pdf_files(
                folder_path, embedding_file_path
            )
            logger.info(f"完成处理，共 {len(documents_with_embeddings)} 个文档块")
            return len(documents_with_embeddings)
        except Exception as e:
            logger.error(f"处理财报文件时出错: {str(e)}")
            raise

    def interactive_mode(self):
        """交互式问答模式"""
        print("\n欢迎使用 RAG 问答系统！")
        print("输入 'quit' 或 'exit' 退出系统")
        print("输入 'help' 查看帮助信息")
        
        while True:
            try:
                # 获取用户输入
                query = input("\n请输入您的问题: ").strip()
                
                # 检查退出命令
                if query.lower() in ['quit', 'exit']:
                    print("感谢使用，再见！")
                    break
                    
                # 检查帮助命令
                if query.lower() == 'help':
                    self.print_help()
                    continue
                    
                # 处理空输入
                if not query:
                    print("请输入有效的问题！")
                    continue
                
                # 处理查询
                print("\n处理中...")
                start_time = time.time()
                
                answer, contexts = self.process_query(query)
                
                # 打印结果
                print(f"\n答案: {answer}")
                print(f"\n处理时间: {time.time() - start_time:.2f} 秒")
                
                # 显示相关上下文
                self.print_contexts(contexts)
                
            except KeyboardInterrupt:
                print("\n程序被中断，正在退出...")
                break
            except Exception as e:
                logger.error(f"发生错误: {str(e)}")
                print("发生错误，请重试！")

    def print_help(self):
        """打印帮助信息"""
        print("\n=== 帮助信息 ===")
        print("1. 输入问题并按回车提交")
        print("2. 系统会返回答案和相关上下文")
        print("3. 输入 'quit' 或 'exit' 退出系统")
        print("4. 输入 'help' 显示此帮助信息")
        print("===============")

    def print_contexts(self, contexts: List[str]):
        """打印相关上下文"""
        if contexts:
            print("\n=== 相关上下文 ===")
            for i, context in enumerate(contexts, 1):
                print(f"\nContext {i}:")
                print(context[:200] + "..." if len(context) > 200 else context)
            print("===============")

    def test_models(self):
        """测试本地模型"""
        logger.info("开始测试本地模型...")
        
        # 测试嵌入模型
        try:
            logger.info("测试嵌入模型...")
            test_text = "这是一个测试句子，用于验证嵌入模型是否工作正常。"
            
            embedding = self.embedding_processor.embed_text(test_text)
            logger.info(f"嵌入向量维度: {len(embedding)}")
            logger.info(f"嵌入向量样本: {embedding[:5]}...")
            logger.info("嵌入模型测试成功")
            
        except Exception as e:
            logger.error(f"嵌入模型测试失败: {str(e)}")
        
        # 测试生成模型
        if self.tokenizer is not None and self.model is not None:
            try:
                logger.info("测试生成模型...")
                test_prompt = "财务报表分析的主要目的是什么?"
                
                inputs = self.tokenizer(
                    test_prompt, 
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH
                ).to(DEVICES)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        num_beams=2,
                        temperature=0.7,
                        early_stopping=True
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"生成模型响应: {response}")
                logger.info("生成模型测试成功")
                
            except Exception as e:
                logger.error(f"生成模型测试失败: {str(e)}")
        else:
            logger.info("未加载本地生成模型，跳过测试")
        
        logger.info("本地模型测试完成")

def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="RAG 系统")
    parser.add_argument(
        '--mode', 
        choices=['train', 'interactive', 'api', 'test_model'], 
        default='interactive',
        help="运行模式：train(训练)、interactive(交互式)、api(API服务)或test_model(模型测试)"
    )
    parser.add_argument(
        '--folder_path', 
        default=FINANCIAL_REPORTS_DIR,
        help="财报PDF文件目录路径"
    )
    parser.add_argument(
        '--embedding_file', 
        default=EMBEDDING_FILE,
        help="嵌入向量存储路径"
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help="API服务端口号"
    )
    parser.add_argument(
        '--use_api', 
        action='store_true',
        help="强制使用OpenAI API"
    )
    parser.add_argument(
        '--use_local', 
        action='store_true',
        help="强制使用本地模型"
    )
    return parser.parse_args()

def main(args=None):
    """主函数"""
    # 设置日志
    setup_logging()
    
    # 解析参数
    if args is None:
        args = parse_args()
    
    try:
        # 根据命令行参数修改配置
        if args.use_api:
            # 强制使用API
            logger.info("根据命令行参数，强制使用OpenAI API")
            MODELS["embedding"]["path"] = None
            MODELS["generation"]["path"] = None
        elif args.use_local:
            # 强制使用本地模型
            logger.info("根据命令行参数，强制使用本地模型")
            if not os.path.exists(MODELS["embedding"]["path"]) or not os.path.exists(MODELS["generation"]["path"]):
                logger.error("本地模型路径不存在，但指定了--use_local参数")
                return 1
        
        # 初始化RAG系统
        logger.info("初始化 RAG 系统...")
        rag_system = RAGSystem(embedding_file=args.embedding_file)
        
        # 根据模式运行
        if args.mode == 'train':
            logger.info("运行训练模式...")
            rag_system.process_financial_reports(
                args.folder_path, args.embedding_file
            )
            logger.info(f"训练完成，嵌入向量已保存到 {args.embedding_file}")
            
        elif args.mode == 'interactive':
            logger.info("运行交互式模式...")
            rag_system.interactive_mode()
            
        elif args.mode == 'api':
            logger.info(f"运行API服务，端口：{args.port}")
            from src.api import run_api_server
            run_api_server(rag_system, port=args.port)
            
        elif args.mode == 'test_model':
            logger.info("运行模型测试模式...")
            rag_system.test_models()
            
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)