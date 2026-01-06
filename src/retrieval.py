
import os
import logging
from typing import List, Tuple, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import numpy as np
from abc import ABC, abstractmethod
import time
import torch

logger = logging.getLogger(__name__)

class RetrievalStrategy(ABC):
    """
    @class RetrievalStrategy
    @description 检索策略基类
    """
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        user_context: Optional[str] = None
    ) -> List[str]:
        """
        检索相关文档
        @param query: 查询文本
        @param top_k: 返回的文档数量
        @param user_context: 用户上下文
        @return: 相关文档列表
        """
        pass

class FactualRetrievalStrategy(RetrievalStrategy):
    """
    @class FactualRetrievalStrategy
    @description 事实检索策略
    """
    def __init__(self, documents_with_embeddings: List[Tuple[str, List[float], str]]):
        self.documents = documents_with_embeddings
        self.embeddings = np.array([doc[1] for doc in documents_with_embeddings])
        self.texts = [doc[0] for doc in documents_with_embeddings]
        
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        user_context: Optional[str] = None
    ) -> List[str]:
        """事实检索"""
        try:
            # 使用 HuggingFace 模型生成查询的嵌入向量
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            query_embedding = embedding_model.embed_query(query)
            
            # 计算相似度
            similarities = np.dot(self.embeddings, query_embedding)
            
            # 使用关键词增强检索结果
            # 检查文档中是否包含关键词
            financial_keywords = ["revenue", "income", "profit", "收入", "利润", "营收", "亿元", "百万"]
            year_keywords = ["2022", "FY22", "FY2022", "财年"]
            
            # 增强包含关键词的文档得分
            keyword_bonus = np.zeros(len(self.texts))
            for i, text in enumerate(self.texts):
                # 检查是否包含财务关键词
                for keyword in financial_keywords:
                    if keyword in text:
                        keyword_bonus[i] += 0.1
                        
                # 检查是否包含年份关键词
                for keyword in year_keywords:
                    if keyword in text:
                        keyword_bonus[i] += 0.05
                        
                # 检查是否同时包含公司名和年份
                if ("Airstar" in text or "airstar" in text.lower()) and any(year in text for year in year_keywords):
                    keyword_bonus[i] += 0.2
                
                # 优先考虑表格数据
                if "[TABLE]" in text and "[/TABLE]" in text:
                    # 如果是表格数据，且包含查询相关的关键词，大幅提升得分
                    if any(kw in text.lower() for kw in query.lower().split()):
                        keyword_bonus[i] += 0.5
                    else:
                        # 即使不直接包含查询词，也适当提升表格的权重
                        keyword_bonus[i] += 0.2
            
            # 将关键词得分加入相似度计算
            adjusted_similarities = similarities + keyword_bonus
            
            # 获取得分最高的文档
            top_indices = np.argsort(adjusted_similarities)[-top_k:][::-1]
            
            # 处理返回结果，确保优先包含表格（如果有的话）
            selected_docs = [self.texts[i] for i in top_indices]
            
            # 检查是否有表格数据
            has_table = any("[TABLE]" in doc for doc in selected_docs)
            
            # 如果没有表格数据，尝试在其他文档中寻找相关表格
            if not has_table and "收入" in query.lower() or "revenue" in query.lower():
                # 找出所有包含表格的文档
                table_docs_indices = [i for i, text in enumerate(self.texts) if "[TABLE]" in text]
                
                if table_docs_indices:
                    # 从表格文档中找出最相关的
                    table_similarities = adjusted_similarities[table_docs_indices]
                    best_table_idx = table_docs_indices[np.argmax(table_similarities)]
                    
                    # 如果最佳表格的相似度不是太低，则将其添加到结果中
                    if adjusted_similarities[best_table_idx] > 0.1:
                        # 确保返回结果不超过 top_k
                        if len(selected_docs) >= top_k:
                            # 替换最不相关的文档
                            min_sim_idx = np.argmin(adjusted_similarities[top_indices])
                            selected_docs[min_sim_idx] = self.texts[best_table_idx]
                        else:
                            selected_docs.append(self.texts[best_table_idx])
            
            return selected_docs
            
        except Exception as e:
            logger.error(f"事实检索时出错: {str(e)}")
            return []

class AnalyticalRetrievalStrategy(RetrievalStrategy):
    """
    @class AnalyticalRetrievalStrategy
    @description 分析检索策略
    """
    def __init__(self, documents_with_embeddings: List[Tuple[str, List[float], str]]):
        self.documents = documents_with_embeddings
        self.embeddings = np.array([doc[1] for doc in documents_with_embeddings])
        self.texts = [doc[0] for doc in documents_with_embeddings]
        
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        user_context: Optional[str] = None
    ) -> List[str]:
        """分析检索"""
        try:
            # 使用 HuggingFace 模型生成查询的嵌入向量
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            query_embedding = embedding_model.embed_query(query)
            
            # 计算相似度
            similarities = np.dot(self.embeddings, query_embedding)
            
            # 考虑文档长度
            lengths = np.array([len(text) for text in self.texts])
            normalized_similarities = similarities / np.sqrt(lengths)
            
            top_indices = np.argsort(normalized_similarities)[-top_k:][::-1]
            
            return [self.texts[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"分析检索时出错: {str(e)}")
            return []

class OpinionRetrievalStrategy(RetrievalStrategy):
    """
    @class OpinionRetrievalStrategy
    @description 观点检索策略
    """
    def __init__(self, documents_with_embeddings: List[Tuple[str, List[float], str]]):
        self.documents = documents_with_embeddings
        self.embeddings = np.array([doc[1] for doc in documents_with_embeddings])
        self.texts = [doc[0] for doc in documents_with_embeddings]
        
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        user_context: Optional[str] = None
    ) -> List[str]:
        """观点检索"""
        try:
            # 使用 HuggingFace 模型生成查询的嵌入向量
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            query_embedding = embedding_model.embed_query(query)
            
            # 计算相似度
            similarities = np.dot(self.embeddings, query_embedding)
            
            # 考虑文档多样性
            selected_indices = []
            remaining_indices = list(range(len(self.texts)))
            
            while len(selected_indices) < top_k and remaining_indices:
                # 选择最相似的文档
                current_similarities = similarities[remaining_indices]
                best_idx = remaining_indices[np.argmax(current_similarities)]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                
                # 移除与已选文档过于相似的文档
                if remaining_indices:
                    selected_embedding = self.embeddings[best_idx]
                    remaining_embeddings = self.embeddings[remaining_indices]
                    similarities_to_selected = np.dot(remaining_embeddings, selected_embedding)
                    too_similar = similarities_to_selected > 0.8
                    remaining_indices = [i for i, too_sim in zip(remaining_indices, too_similar) if not too_sim]
            
            return [self.texts[i] for i in selected_indices]
            
        except Exception as e:
            logger.error(f"观点检索时出错: {str(e)}")
            return []

class ContextualRetrievalStrategy(RetrievalStrategy):
    """
    @class ContextualRetrievalStrategy
    @description 上下文检索策略
    """
    def __init__(self, documents_with_embeddings: List[Tuple[str, List[float], str]]):
        self.documents = documents_with_embeddings
        self.embeddings = np.array([doc[1] for doc in documents_with_embeddings])
        self.texts = [doc[0] for doc in documents_with_embeddings]
        
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        user_context: Optional[str] = None
    ) -> List[str]:
        """上下文检索"""
        try:
            # 使用 HuggingFace 模型生成查询的嵌入向量
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # 添加重试机制
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    query_embedding = embedding_model.embed_query(query)
                    
                    # 如果有用户上下文，也生成其嵌入向量
                    if user_context:
                        context_embedding = embedding_model.embed_query(user_context)
                        # 结合查询和上下文的嵌入向量
                        combined_embedding = (query_embedding + context_embedding) / 2
                    else:
                        combined_embedding = query_embedding
                    
                    # 计算相似度
                    similarities = np.dot(self.embeddings, combined_embedding)
                    top_indices = np.argsort(similarities)[-top_k:][::-1]
                    
                    return [self.texts[i] for i in top_indices]
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"第 {retry_count} 次尝试检索失败: {str(e)}")
                    
                    if retry_count < max_retries:
                        # 指数退避策略
                        sleep_time = 2 ** retry_count
                        logger.info(f"等待 {sleep_time} 秒后重试...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"上下文检索失败，已达到最大重试次数: {str(e)}")
                        return []
            
        except Exception as e:
            logger.error(f"上下文检索时出错: {str(e)}")
            return []

class DocumentRetriever:
    """
    @class DocumentRetriever
    @description 文档检索器
    """
    def __init__(self, embedding_file: Optional[str] = None):
        """
        初始化文档检索器
        @param embedding_file: 嵌入向量文件路径
        """
        self.documents_with_embeddings = []
        self.strategy = None
        
        if embedding_file and os.path.exists(embedding_file):
            try:
                with open(embedding_file, 'rb') as f:
                    self.documents_with_embeddings = pickle.load(f)
                logger.info(f"已加载 {len(self.documents_with_embeddings)} 个文档嵌入向量")
                
                # 默认使用事实检索策略
                self.change_strategy("factual")
                
            except Exception as e:
                logger.error(f"加载嵌入向量文件时出错: {str(e)}")
                raise
    
    def change_strategy(self, strategy_name: str):
        """
        切换检索策略
        @param strategy_name: 策略名称
        """
        strategy_map = {
            "factual": FactualRetrievalStrategy,
            "analytical": AnalyticalRetrievalStrategy,
            "opinion": OpinionRetrievalStrategy,
            "contextual": ContextualRetrievalStrategy
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"不支持的检索策略: {strategy_name}")
            
        self.strategy = strategy_map[strategy_name](self.documents_with_embeddings)
        logger.info(f"已切换到 {strategy_name} 检索策略")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        strategy: Optional[str] = None,
        user_context: Optional[str] = None
    ) -> List[str]:
        """
        检索相关文档
        @param query: 查询文本
        @param top_k: 返回的文档数量
        @param strategy: 检索策略
        @param user_context: 用户上下文
        @return: 相关文档列表
        """
        if not self.documents_with_embeddings:
            logger.warning("没有可用的文档嵌入向量")
            return []
            
        if strategy and strategy != self.strategy.__class__.__name__.lower().replace('retrievalstrategy', ''):
            self.change_strategy(strategy)
            
        return self.strategy.retrieve(query, top_k, user_context)
