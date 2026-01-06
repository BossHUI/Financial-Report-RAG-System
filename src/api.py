
import os
import time
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., title="用户查询", description="用户问题或查询文本")
    top_k: int = Field(3, title="返回文档数", description="要返回的相关文档数量")
    strategy: str = Field("factual", title="检索策略", description="检索策略类型")
    user_context: Optional[str] = Field(None, title="用户上下文", description="用于上下文检索的附加信息")

class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(..., title="回答", description="生成的回答")
    contexts: List[str] = Field(..., title="上下文", description="检索的相关上下文")
    process_time: float = Field(..., title="处理时间", description="处理请求的时间（秒）")
    source_files: List[str] = Field(..., title="来源文件", description="上下文来源文件列表")

class StatusResponse(BaseModel):
    """状态响应模型"""
    status: str = Field(..., title="状态", description="系统状态")
    document_count: int = Field(..., title="文档数量", description="系统中的文档数量")
    sources: Dict[str, int] = Field(..., title="来源分布", description="按来源文件分组的文档数量")

def create_app(rag_system):
    """创建FastAPI应用"""
    app = FastAPI(
        title="财报RAG系统API",
        description="通过API访问财报检索增强生成系统",
        version="1.0.0"
    )
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 后台任务：处理财报文件
    def process_financial_reports_task(folder_path, embedding_file_path):
        try:
            rag_system.process_financial_reports(folder_path, embedding_file_path)
            logger.info(f"财报处理完成，文件保存到 {embedding_file_path}")
        except Exception as e:
            logger.error(f"财报处理失败: {str(e)}")
    
    @app.get("/", response_model=StatusResponse)
    async def get_status():
        """获取系统状态"""
        doc_retriever = rag_system.document_retriever
        docs = doc_retriever.documents_with_embeddings
        
        # 按来源分组统计
        sources = {}
        for _, _, source in docs:
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
                
        return StatusResponse(
            status="正常运行中",
            document_count=len(docs),
            sources=sources
        )
    
    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """查询RAG系统"""
        try:
            start_time = time.time()
            
            # 检索文档
            contexts = rag_system.document_retriever.retrieve(
                request.query,
                top_k=request.top_k,
                strategy=request.strategy,
                user_context=request.user_context
            )
            
            if not contexts:
                raise HTTPException(
                    status_code=404, 
                    detail="未找到相关信息。请尝试调整问题或检索策略。"
                )
            
            # 生成答案
            answer = rag_system.generate_answer(request.query, contexts)
            
            # 获取来源文件
            source_files = []
            for i, ctx in enumerate(contexts):
                idx = rag_system.document_retriever.documents_with_embeddings.index(
                    next(
                        doc for doc in rag_system.document_retriever.documents_with_embeddings 
                        if doc[0] == ctx
                    )
                )
                source = rag_system.document_retriever.documents_with_embeddings[idx][2]
                source_files.append(source)
            
            process_time = time.time() - start_time
            
            return QueryResponse(
                answer=answer,
                contexts=contexts,
                process_time=process_time,
                source_files=source_files
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"处理查询时出错: {str(e)}"
            )
    
    @app.post("/process_reports")
    async def process_reports(
        background_tasks: BackgroundTasks,
        folder_path: str = Query(..., description="财报文件夹路径"),
        embedding_file: Optional[str] = Query(
            None, description="嵌入向量存储路径"
        )
    ):
        """处理财报文件并生成嵌入向量（异步任务）"""
        from src.config import FINANCIAL_REPORTS_DIR, EMBEDDING_FILE
        
        try:
            # 使用默认值
            if not folder_path:
                folder_path = FINANCIAL_REPORTS_DIR
            if not embedding_file:
                embedding_file = EMBEDDING_FILE
                
            # 添加后台任务
            background_tasks.add_task(
                process_financial_reports_task, 
                folder_path, 
                embedding_file
            )
            
            return {"status": "success", "message": "财报处理任务已启动"}
            
        except Exception as e:
            logger.error(f"启动财报处理任务时出错: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"启动财报处理任务时出错: {str(e)}"
            )
    
    return app

def run_api_server(rag_system, host="0.0.0.0", port=8000):
    """运行API服务器"""
    app = create_app(rag_system)
    uvicorn.run(app, host=host, port=port) 