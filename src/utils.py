import os
import time
import logging
import logging.handlers
import dotenv
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import io

from src.config import LOG_DIR, MAX_RETRIES, RETRY_DELAY, API_RATE_LIMIT, TABLES_DIR

# 用于API请求限制的变量
_last_request_times = []

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    设置日志
    @param log_file: 日志文件路径
    @param level: 日志级别
    """
    if log_file is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"rag_{time.strftime('%Y%m%d')}.log")
        
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理程序
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # 创建格式化程序
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # 控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # 文件处理程序
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)
    
    # 临时禁用该错误
    logging.getLogger('numexpr').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.info(f"日志已配置，文件: {log_file}")

def load_env_variables() -> Dict[str, Any]:
    """
    加载环境变量
    @return: 环境变量字典
    """
    # 加载.env文件
    dotenv.load_dotenv()
    
    # 获取OpenAI API密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # 检查是否存在有效的API密钥
    if not openai_api_key:
        # 尝试从系统环境变量获取
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        # 如果仍然没有找到
        if not openai_api_key:
            logging.warning("未找到OPENAI_API_KEY环境变量。如果使用OpenAI API，请确保设置此变量。")
            logging.info("如果使用本地模型，可以忽略此警告。")
        else:
            logging.info("从系统环境变量获取到OPENAI_API_KEY")
    else:
        logging.info("从.env文件获取到OPENAI_API_KEY")
    
    # 设置为环境变量，确保OpenAI客户端可以使用
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # 返回环境变量字典
    return {
        "openai_api_key": openai_api_key
    }

def rate_limit_api_calls() -> None:
    """
    实现API调用速率限制
    保持每10秒最多API_RATE_LIMIT个请求
    """
    global _last_request_times
    
    current_time = time.time()
    # 移除10秒前的请求记录
    _last_request_times = [t for t in _last_request_times if current_time - t < 10]
    
    # 检查是否超出速率限制
    if len(_last_request_times) >= API_RATE_LIMIT:
        # 计算需要等待的时间
        wait_time = 10 - (current_time - _last_request_times[0])
        if wait_time > 0:
            logging.info(f"API速率限制: 等待 {wait_time:.2f} 秒")
            time.sleep(wait_time)
    
    # 添加当前请求时间
    _last_request_times.append(time.time())

def retry_with_exponential_backoff(func, *args, **kwargs):
    """
    使用指数退避策略的重试装饰器
    @param func: 要执行的函数
    @param args: 位置参数
    @param kwargs: 关键字参数
    @return: 函数执行结果
    """
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # 应用API速率限制
            rate_limit_api_calls()
            
            # 执行函数
            return func(*args, **kwargs)
            
        except Exception as e:
            retry_count += 1
            if retry_count < MAX_RETRIES:
                # 计算退避时间
                sleep_time = RETRY_DELAY * (2 ** retry_count)
                logging.warning(f"第 {retry_count} 次重试失败: {str(e)}，等待 {sleep_time} 秒后重试...")
                time.sleep(sleep_time)
            else:
                logging.error(f"重试失败，已达到最大重试次数: {str(e)}")
                raise

# 表格相关工具函数
def dataframe_to_html(df: pd.DataFrame, max_rows: int = 20, max_cols: int = 10) -> str:
    """
    将DataFrame转换为HTML表格
    @param df: DataFrame对象
    @param max_rows: 最大行数
    @param max_cols: 最大列数
    @return: HTML表格
    """
    # 处理大型表格
    if len(df) > max_rows or len(df.columns) > max_cols:
        # 截取表格
        display_df = df.iloc[:max_rows, :max_cols]
        
        if len(df) > max_rows:
            # 添加省略号行
            ellipsis_row = pd.Series(['...'] * len(display_df.columns), index=display_df.columns)
            display_df = pd.concat([display_df, pd.DataFrame([ellipsis_row])], ignore_index=True)
            
        if len(df.columns) > max_cols:
            # 添加省略号列
            display_df['...'] = '...'
            
        html = display_df.to_html(classes='table table-striped table-hover')
        html += f'<p><em>显示 {max_rows}/{len(df)} 行 和 {max_cols}/{len(df.columns)} 列</em></p>'
    else:
        html = df.to_html(classes='table table-striped table-hover')
        
    return html

def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20, max_cols: int = 10) -> str:
    """
    将DataFrame转换为Markdown表格
    @param df: DataFrame对象
    @param max_rows: 最大行数
    @param max_cols: 最大列数
    @return: Markdown表格
    """
    # 处理大型表格
    if len(df) > max_rows or len(df.columns) > max_cols:
        # 截取表格
        display_df = df.iloc[:max_rows, :max_cols]
        
        if len(df) > max_rows:
            # 添加省略号行
            ellipsis_row = pd.Series(['...'] * len(display_df.columns), index=display_df.columns)
            display_df = pd.concat([display_df, pd.DataFrame([ellipsis_row])], ignore_index=True)
            
        if len(df.columns) > max_cols:
            # 添加省略号列
            display_df['...'] = '...'
            
        markdown = display_df.to_markdown()
        markdown += f'\n\n*显示 {max_rows}/{len(df)} 行 和 {max_cols}/{len(df.columns)} 列*'
    else:
        markdown = df.to_markdown()
        
    return markdown

def format_table_for_display(table_info: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
    """
    格式化表格信息用于显示
    @param table_info: 表格信息字典
    @param df: 表格数据
    @return: 格式化的表格信息
    """
    output = []
    
    # 添加表格标识
    output.append(f"表格ID: {table_info.get('id', 'Unknown')}")
    output.append(f"来源: {table_info.get('source', 'Unknown')}")
    output.append(f"页码: {table_info.get('page', 'Unknown')}")
    
    # 如果有自定义表格数据，使用它替代表格信息中的数据
    if df is not None and not df.empty:
        output.append("\n表格数据:")
        output.append(dataframe_to_markdown(df))
    elif 'csv' in table_info:
        # 从CSV恢复DataFrame
        try:
            csv_df = pd.read_csv(io.StringIO(table_info['csv']))
            output.append("\n表格数据:")
            output.append(dataframe_to_markdown(csv_df))
        except Exception as e:
            logging.error(f"无法解析CSV数据: {str(e)}")
            output.append("\n原始表格文本:")
            output.append(f"```\n{table_info.get('text', '')}\n```")
    else:
        # 使用原始文本
        output.append("\n原始表格文本:")
        output.append(f"```\n{table_info.get('text', '')}\n```")
    
    return "\n".join(output)

def list_extracted_tables() -> List[str]:
    """
    列出已提取的表格
    @return: 表格ID列表
    """
    if not os.path.exists(TABLES_DIR):
        return []
        
    return [f.replace('.pkl', '') for f in os.listdir(TABLES_DIR) if f.endswith('.pkl')]

def get_table_summary(tables: List[Dict[str, Any]]) -> str:
    """
    获取表格摘要信息
    @param tables: 表格列表
    @return: 表格摘要
    """
    if not tables:
        return "未找到表格"
        
    summary = []
    summary.append(f"共找到 {len(tables)} 个表格:")
    
    for i, table in enumerate(tables, 1):
        table_id = table.get('id', f'表格{i}')
        source = os.path.basename(table.get('source', 'Unknown'))
        page = table.get('page', 'Unknown')
        
        # 计算表格文本长度
        text_length = len(table.get('text', ''))
        
        # 获取表格列名
        columns = table.get('columns', [])
        columns_str = ", ".join(columns[:5])
        if len(columns) > 5:
            columns_str += f" ... 等共{len(columns)}列"
            
        summary.append(f"{i}. {table_id} - 来源: {source}, 页码: {page}, 长度: {text_length} 字符")
        if columns:
            summary.append(f"   列: {columns_str}")
            
    return "\n".join(summary)
