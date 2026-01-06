
import os
import logging
from typing import List, Tuple, Optional
import torch
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer
from PyPDF2 import PdfReader
import pickle
from tqdm import tqdm
import time
import numpy as np
import re
from src.config import REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY, MODELS, DEVICES, API_RATE_LIMIT

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """
    @class EmbeddingProcessor
    @description 文档嵌入处理类
    """
    def __init__(self):
        """初始化嵌入处理器"""
        try:
            # 检查是否使用本地模型
            model_path = MODELS["embedding"]["path"]
            model_name = MODELS["embedding"]["name"]
            
            # 初始化为None
            self.embedding_model = None
            self.use_local_model = False
            
            # 尝试加载本地模型
            if model_path and os.path.exists(model_path):
                logger.info(f"尝试加载本地嵌入模型: {model_path}")
                
                try:
                    # 检查模型文件是否存在
                    model_files = os.listdir(model_path)
                    logger.info(f"发现模型文件: {', '.join(model_files)}")
                    
                    # 使用HuggingFace模型
                    self.embedding_model = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={"device": DEVICES},
                        encode_kwargs={"normalize_embeddings": True}
                    )
                    self.use_local_model = True
                    logger.info("成功加载本地嵌入模型")
                    
                except Exception as e:
                    logger.error(f"加载本地嵌入模型失败: {str(e)}")
                    logger.info("将回退至OpenAI API")
                    self.use_local_model = False
            
            # 如果本地模型加载失败或不存在，使用OpenAI API
            if not self.use_local_model:
                logger.info(f"使用OpenAI嵌入模型: {model_name}")
                
                # 使用OpenAI API
                self.embedding_model = OpenAIEmbeddings(
                    model=model_name,
                    chunk_size=1000,
                    request_timeout=REQUEST_TIMEOUT
                )
                
            logger.info("嵌入模型初始化完成")
            
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {str(e)}")
            raise
    
    def detect_table(self, text: str) -> bool:
        """
        检测文本中是否包含表格
        @param text: 文本内容
        @return: 是否包含表格
        """
        # 表格检测特征
        table_indicators = [
            # 表格行模式（多个连续数字或分隔符）
            r'(\d+[\s,\.]+){3,}',  # 多个连续数字
            r'[\|\+][-\s]*[\|\+]',  # ASCII 表格分隔符
            r'(\s{2,}\S+){3,}',     # 多个由空格分隔的元素
            # 表格标题特征
            r'(revenue|income|profit|statement|balance|sheet|assets|liabilities|equity|年|收入|利润|资产|负债|权益)\s*[\:\：]?.*?(\d{4}|\d{2}\/\d{2})',
            # 表格行列计数模式
            r'(total|sum|小计|合计|总计)',
            # 电子表格参考
            r'[A-Z]\d+[\:\：][A-Z]\d+'
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False
    
    def is_table_continuation(self, prev_text: str, curr_text: str) -> bool:
        """
        检查当前文本是否是前一个表格的延续
        @param prev_text: 前一段文本
        @param curr_text: 当前文本
        @return: 是否是表格延续
        """
        # 表格延续特征
        # 1. 检查数字格式是否相似
        prev_numbers = re.findall(r'\d+[,\.]?\d*', prev_text)
        curr_numbers = re.findall(r'\d+[,\.]?\d*', curr_text)
        
        if len(prev_numbers) >= 3 and len(curr_numbers) >= 3:
            return True
            
        # 2. 检查分隔符模式
        prev_separators = re.findall(r'[\|\+]', prev_text)
        curr_separators = re.findall(r'[\|\+]', curr_text)
        
        if len(prev_separators) >= 3 and len(curr_separators) >= 3:
            return True
            
        # 3. 检查空格对齐模式
        prev_spaces = [len(s) for s in re.findall(r'\s+', prev_text)]
        curr_spaces = [len(s) for s in re.findall(r'\s+', curr_text)]
        
        if len(prev_spaces) >= 3 and len(curr_spaces) >= 3:
            similarities = sum(1 for p, c in zip(prev_spaces, curr_spaces) if abs(p-c) <= 1)
            if similarities >= 3:
                return True
                
        return False
    
    def embed_text(self, text: str) -> List[float]:
        """
        使用嵌入模型生成文本的嵌入向量
        @param text: 要嵌入的文本
        @return: 嵌入向量
        """
        if self.use_local_model:
            # 本地模型嵌入处理
            max_retries = 2  # 本地模型少量重试
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    embedding = self.embedding_model.embed_query(text)
                    return embedding
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"本地模型生成嵌入向量失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                    
                    if retry_count < max_retries:
                        logger.info("等待1秒后重试...")
                        time.sleep(1)
                    else:
                        logger.error(f"本地模型生成嵌入向量失败，达到最大重试次数: {str(e)}")
                        raise
        else:
            # OpenAI API嵌入处理，带重试逻辑
            max_retries = MAX_RETRIES
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # 在每次请求前添加速率限制
                    self._rate_limit_api_call()
                    
                    embedding = self.embedding_model.embed_query(text)
                    return embedding
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"OpenAI API生成嵌入向量失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                    
                    if retry_count < max_retries:
                        sleep_time = RETRY_DELAY * (2 ** retry_count)
                        logger.info(f"等待 {sleep_time} 秒后重试...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"OpenAI API生成嵌入向量失败，达到最大重试次数: {str(e)}")
                        raise
    
    def _rate_limit_api_call(self):
        """应用API调用速率限制"""
        global _last_request_times
        
        if not hasattr(self, '_last_request_times'):
            self._last_request_times = []
            
        current_time = time.time()
        # 移除10秒前的请求记录
        self._last_request_times = [t for t in self._last_request_times if current_time - t < 10]
        
        # 检查是否超出速率限制
        if len(self._last_request_times) >= API_RATE_LIMIT:
            # 计算需要等待的时间
            wait_time = 10 - (current_time - self._last_request_times[0])
            if wait_time > 0:
                logger.info(f"API速率限制: 等待 {wait_time:.2f} 秒...")
                time.sleep(wait_time)
        
        # 添加当前请求时间
        self._last_request_times.append(time.time())

    def process_pdf_files(
        self, 
        folder_path: str, 
        embedding_file: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Tuple[str, List[float], str]]:
        """
        处理PDF文件并生成嵌入向量
        @param folder_path: PDF文件目录路径
        @param embedding_file: 嵌入向量存储文件路径
        @param chunk_size: 文本块大小
        @param chunk_overlap: 文本块重叠大小
        @return: 文档块及其嵌入向量列表
        """
        try:
            # 创建存储嵌入向量的目录(如果不存在)
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            
            # 加载已存在的嵌入向量
            existing_embeddings = []
            if os.path.exists(embedding_file):
                with open(embedding_file, 'rb') as f:
                    existing_embeddings = pickle.load(f)
                    logger.info(f"已加载 {len(existing_embeddings)} 个现有嵌入向量")

            # 获取已处理的文件列表
            processed_files = set()
            for _, _, source in existing_embeddings:
                processed_files.add(source)

            # 文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            # 处理新文件
            new_embeddings = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if not file.lower().endswith('.pdf'):
                        continue
                        
                    file_path = os.path.join(root, file)
                    if file_path in processed_files:
                        logger.info(f"跳过已处理的文件: {file_path}")
                        continue
                        
                    try:
                        logger.info(f"处理文件: {file_path}")
                        
                        try:
                            reader = PdfReader(file_path)
                            
                            # 提取文本
                            text = ""
                            page_texts = []
                            for page_num, page in enumerate(reader.pages):
                                try:
                                    page_text = page.extract_text()
                                    if page_text:
                                        page_texts.append(page_text)
                                        text += page_text + "\n\n"
                                    else:
                                        logger.warning(f"页面 {page_num} 文本提取为空")
                                except Exception as e:
                                    logger.error(f"处理 {file_path} 页面 {page_num} 时出错: {str(e)}")
                            
                            # 检查提取的文本是否为空
                            if not text.strip():
                                logger.warning(f"从 {file_path} 提取的文本为空，可能是扫描版PDF或格式问题")
                                # 尝试使用其他方法提取（示例）
                                continue
                                
                            # 记录文本长度，帮助调试
                            logger.info(f"从 {file_path} 提取了 {len(text)} 字符的文本")
                            
                            # 特殊处理：在分割前检测和处理表格
                            # 1. 首先检测各页中的表格
                            table_sections = []
                            in_table = False
                            current_table = ""
                            
                            for i, page_text in enumerate(page_texts):
                                # 分析页面文本
                                for line in page_text.split('\n'):
                                    # 检测是否是表格行
                                    if self.detect_table(line):
                                        if not in_table:
                                            in_table = True
                                            current_table = line + "\n"
                                        else:
                                            current_table += line + "\n"
                                    elif in_table and self.is_table_continuation(current_table, line):
                                        # 检测表格延续
                                        current_table += line + "\n"
                                    elif in_table:
                                        # 表格结束
                                        if len(current_table.strip()) > 100:  # 确保表格内容足够多
                                            logger.info(f"检测到表格，长度: {len(current_table)} 字符")
                                            table_sections.append((current_table, file_path))
                                        in_table = False
                                        current_table = ""
                                
                            # 处理最后一个未完成的表格
                            if in_table and len(current_table.strip()) > 100:
                                logger.info(f"检测到表格，长度: {len(current_table)} 字符")
                                table_sections.append((current_table, file_path))
                                
                            # 分割文本
                            chunks = text_splitter.split_text(text)
                            logger.info(f"文件 {file} 被分割为 {len(chunks)} 个文本块")
                            
                            # 生成嵌入向量
                            for i, chunk in enumerate(tqdm(chunks, desc=f"处理 {file}")):
                                # 添加延迟，避免触发API速率限制
                                if not self.use_local_model and i > 0 and i % 2 == 0:  # 每2个请求暂停
                                    sleep_time = 3  # 暂停3秒
                                    logger.info(f"API速率限制：暂停 {sleep_time} 秒...")
                                    time.sleep(sleep_time)
                                
                                try:
                                    # 确保文本块有意义
                                    if len(chunk.strip()) < 20:  # 忽略过短的文本块
                                        continue
                                        
                                    embedding = self.embed_text(chunk)
                                    new_embeddings.append((chunk, embedding, file_path))
                                except Exception as e:
                                    logger.error(f"处理文本块失败: {str(e)}")
                                    continue
                            
                            # 处理表格部分
                            for table_text, source in table_sections:
                                try:
                                    # 单独处理每个表格部分
                                    logger.info(f"处理表格文本，长度: {len(table_text)}")
                                    embedding = self.embed_text(table_text)
                                    # 添加特殊标记，表明这是表格
                                    marked_table = f"[TABLE]\n{table_text}\n[/TABLE]"
                                    new_embeddings.append((marked_table, embedding, source))
                                except Exception as e:
                                    logger.error(f"处理表格文本失败: {str(e)}")
                                    continue
                                    
                        except Exception as e:
                            logger.error(f"读取 PDF 文件 {file_path} 失败: {str(e)}")
                            continue
                        
                        # 定期保存进度，避免处理大量文件时丢失进度
                        if len(new_embeddings) >= 50:
                            all_embeddings = existing_embeddings + new_embeddings
                            with open(embedding_file, 'wb') as f:
                                pickle.dump(all_embeddings, f)
                            logger.info(f"已保存中间进度，当前共 {len(all_embeddings)} 个嵌入向量")
                            existing_embeddings = all_embeddings
                            new_embeddings = []
                            
                    except Exception as e:
                        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                        continue

            # 合并新旧嵌入向量
            all_embeddings = existing_embeddings + new_embeddings
            
            # 保存嵌入向量
            if new_embeddings:
                with open(embedding_file, 'wb') as f:
                    pickle.dump(all_embeddings, f)
                logger.info(f"保存了 {len(new_embeddings)} 个新的嵌入向量")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"处理PDF文件时出错: {str(e)}")
            raise