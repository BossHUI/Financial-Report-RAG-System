# -*- coding: utf-8 -*-

import os
import time
import gradio as gr
import pickle
from typing import List, Tuple, Dict
import asyncio
import pandas as pd
import re
import io
from openai import OpenAI
import logging

from src.config import FINANCIAL_REPORTS_DIR, EMBEDDING_FILE, MODELS, DEVICES, TABLES_DIR, TABLE_INDEX_FILE
from src.embeddings import EmbeddingProcessor
from src.retrieval import DocumentRetriever
from src.utils import load_env_variables, dataframe_to_html, format_table_for_display, get_table_summary
from src.main import RAGSystem
from src.table_extractor import TableExtractor

# 全局变量
rag_system = None
chat_history = []
client = None
table_extractor = None
logger = logging.getLogger(__name__)

# 加载环境变量
try:
    env_vars = load_env_variables()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    raise Exception(f"加载环境变量失败: {str(e)}\n请确保项目根目录中存在.env文件，且包含有效的OPENAI_API_KEY")

def initialize_rag_system():
    """初始化RAG系统"""
    global rag_system, table_extractor
    try:
        rag_system = RAGSystem(embedding_file=EMBEDDING_FILE)
        # 初始化表格提取器
        table_extractor = TableExtractor()
        return True
    except Exception as e:
        return False

def process_financial_reports(progress=gr.Progress()):
    """处理财报文件并生成嵌入向量"""
    try:
        # 初始化嵌入处理器
        embedding_processor = EmbeddingProcessor()
        
        # 处理PDF文件
        progress(0, desc="正在处理财报文件...")
        result = embedding_processor.process_pdf_files(
            FINANCIAL_REPORTS_DIR, EMBEDDING_FILE
        )
        progress(1, desc="处理完成")
        
        if result:
            return f"成功处理 {len(result)} 个文档块！"
        else:
            return "没有新文件需要处理"
            
    except Exception as e:
        return f"处理财报文件时出错: {str(e)}"

def reprocess_financial_reports(progress=gr.Progress()):
    """强制重新处理所有财报文件"""
    try:
        # 删除现有的嵌入向量文件
        if os.path.exists(EMBEDDING_FILE):
            os.remove(EMBEDDING_FILE)
            
        # 初始化嵌入处理器
        embedding_processor = EmbeddingProcessor()
        
        # 处理PDF文件
        progress(0, desc="正在重新处理所有财报文件...")
        result = embedding_processor.process_pdf_files(
            FINANCIAL_REPORTS_DIR, EMBEDDING_FILE
        )
        progress(1, desc="处理完成")
        
        # 重新初始化RAG系统
        global rag_system
        rag_system = RAGSystem(embedding_file=EMBEDDING_FILE)
        
        return f"成功重新处理 {len(result)} 个文档块！"
            
    except Exception as e:
        return f"重新处理财报文件时出错: {str(e)}"

# 新增表格提取功能
def extract_tables(progress=gr.Progress()):
    """从所有财报中提取表格"""
    global table_extractor
    try:
        if not table_extractor:
            table_extractor = TableExtractor()
            
        # 提取表格
        progress(0, desc="正在从财报中提取表格...")
        results = table_extractor.extract_tables_from_folder(FINANCIAL_REPORTS_DIR)
        
        # 计算总表格数
        total_tables = sum(len(tables) for tables in results.values())
        progress(1, desc="表格提取完成")
        
        # 生成摘要
        if total_tables > 0:
            all_tables = []
            for tables in results.values():
                all_tables.extend(tables)
                
            summary = get_table_summary(all_tables)
            return f"成功从 {len(results)} 个文件中提取 {total_tables} 个表格！\n\n{summary}"
        else:
            return "未能从财报中提取到任何表格。"
            
    except Exception as e:
        return f"提取表格时出错: {str(e)}"

def search_tables(query: str) -> Tuple[str, pd.DataFrame]:
    """根据关键词搜索表格"""
    global table_extractor
    try:
        if not table_extractor:
            return "表格提取器未初始化，请先提取表格。", None
            
        if not query.strip():
            return "请输入搜索关键词", None
            
        # 搜索表格
        tables = table_extractor.search_tables(query, top_k=5)
        
        if not tables:
            return f"未找到与 '{query}' 相关的表格。", None
            
        # 生成摘要
        summary = get_table_summary(tables)
        
        # 获取第一个表格的数据
        if tables:
            first_table = tables[0]
            df = table_extractor.get_table_by_id(first_table['id'])
            
            if df is not None:
                # 添加表格来源标题行
                source = first_table.get('source', '').split('/')[-1]
                page = first_table.get('page', '未知')
                title = f"表格来源: {source} (第{page}页)"
                
                # 表格描述信息
                table_info = f"{summary}\n\n【当前表格】来源: {source}, 页码: {page}, 行数: {len(df)}, 列数: {len(df.columns)}"
                
                # 为表格添加元数据标题行
                meta_df = pd.DataFrame([{'表格信息': title}])
                result_df = pd.concat([meta_df, df], ignore_index=False)
                
                # 格式化表格显示
                result_df = format_table_display(result_df)
                
                return table_info, result_df
            else:
                return f"{summary}\n\n无法加载表格数据。", None
        
        return summary, None
            
    except Exception as e:
        return f"搜索表格时出错: {str(e)}", None

def preprocess_financial_context(context: str) -> str:
    """
    预处理财务上下文，识别并标记财务数据
    """
    lines = context.split('\n')
    processed_lines = []
    
    # 财务数据正则表达式
    money_pattern = r'(\d+[\d,]*\.?\d*)\s*(million|billion|trillion|百万|亿|兆)'
    percentage_pattern = r'(\d+\.?\d*)\s*(%|percent|百分点)'
    year_pattern = r'(FY)?\s*(20\d{2})(\s*财年)?'
    
    for line in lines:
        # 识别并标记金额
        line = re.sub(money_pattern, r'【金额: \1 \2】', line)
        
        # 识别并标记百分比
        line = re.sub(percentage_pattern, r'【百分比: \1%】', line)
        
        # 识别并标记年份
        line = re.sub(year_pattern, r'【年份: \2】', line)
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def detect_and_format_table(text: str) -> Tuple[bool, str]:
    """
    检测文本中的表格并格式化
    @param text: 原始文本
    @return: (是否包含表格, 格式化后的表格文本)
    """
    # 表格检测模式
    has_table = False
    formatted_table = ""
    
    # 检测表格的几种常见模式
    table_patterns = [
        # 包含多个数字列和行标题的模式
        r'(\|[-+]+\|)+',  # Markdown表格分隔符
        r'(\+[-+]+\+)+',  # ASCII表格分隔符
        r'((\d+[.,]?\d*\s*){3,})',  # 连续多个数字
        r'(([\w\s]+\s*\d+[.,]?\d*\s*){3,})',  # 标题+数字模式
        r'\b([\w\s]+)\s+(\d{4})\s+(\d{4})',  # 项目名+年份+年份模式(常见于财报)
        r'(\$\s*\d+[.,]?\d*\s*){2,}',  # 多个金额
        r'(人民币\s*\d+[.,]?\d*\s*){2,}',  # 中文金额
        r'(¥\s*\d+[.,]?\d*\s*){2,}'  # 人民币符号+金额
    ]
    
    # 检查是否匹配表格模式
    for pattern in table_patterns:
        if re.search(pattern, text):
            has_table = True
            break
    
    if has_table:
        # 尝试格式化表格
        lines = text.split('\n')
        cleaned_lines = []
        in_table = False
        table_rows = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                continue
                
            # 检测表格行：包含多个数字、包含分隔符或者有对齐的空格
            cells = re.findall(r'([^\|\+]+)', line)
            numeric_cells = sum(1 for cell in cells if re.search(r'\d+', cell))
            has_delimiters = '|' in line or '+' in line or '\t' in line
            has_aligned_spaces = bool(re.search(r'\s{2,}[\w\d]+', line))
            
            # 如果一行中有多个数字单元格或分隔符，可能是表格
            if numeric_cells >= 2 or has_delimiters or has_aligned_spaces:
                in_table = True
                table_rows.append(line)
            elif in_table and len(table_rows) < 3:
                # 如果表格太短可能不是真正的表格，继续收集
                table_rows.append(line)
            elif in_table:
                # 如果之前在表格中，这里不再是表格行，结束表格
                if len(table_rows) >= 3:  # 至少需要3行才算表格
                    # 处理收集到的表格行
                    formatted_table = "\n".join(table_rows)
                    break
                in_table = False
                table_rows = []
        
        # 处理最后一个表格
        if in_table and len(table_rows) >= 3:
            formatted_table = "\n".join(table_rows)
            
        # 如果没有成功格式化，返回原文本中可能的表格部分
        if not formatted_table:
            # 尝试提取最长的连续数字行块
            numeric_blocks = []
            current_block = []
            
            for line in lines:
                if re.search(r'\d+', line) and len(line.strip()) > 10:  # 排除太短的行
                    current_block.append(line)
                elif current_block:
                    if len(current_block) >= 3:  # 至少三行才算块
                        numeric_blocks.append(current_block)
                    current_block = []
                    
            if current_block and len(current_block) >= 3:
                numeric_blocks.append(current_block)
                
            # 选择最长的数字块
            if numeric_blocks:
                best_block = max(numeric_blocks, key=len)
                formatted_table = "\n".join(best_block)
            else:
                formatted_table = text  # 无法识别表格，返回原文
    
    return has_table, formatted_table

def convert_text_to_dataframe(text: str) -> pd.DataFrame:
    """
    尝试将文本转换为DataFrame
    @param text: 表格文本
    @return: DataFrame或None
    """
    try:
        # 尝试不同方法解析表格
        df = None
        text = text.strip()
        
        # 方法1: 如果是类似CSV格式(使用|或,分隔)
        if '|' in text:
            try:
                # 删除表格装饰字符，但保留|作为分隔符
                cleaned_text = re.sub(r'[\+\-=]', '', text)
                df = pd.read_csv(io.StringIO(cleaned_text), sep='|', skipinitialspace=True)
                # 清理列名(删除两边的空白和空列)
                df.columns = [str(col).strip() for col in df.columns]
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                if not df.empty:
                    return df
            except Exception:
                pass
        
        # 方法2: 固定宽度格式
        try:
            df = pd.read_fwf(io.StringIO(text))
            if not df.empty and len(df.columns) > 1:
                return df
        except Exception:
            pass
        
        # 方法3: 空格分隔格式 - 尝试识别基于空格的对齐
        try:
            lines = text.split('\n')
            if len(lines) >= 2:
                # 分析第一行中的空格位置
                header = lines[0]
                space_positions = [m.start() for m in re.finditer(r'\s{2,}', header)]
                
                if space_positions:
                    # 基于空格位置分割所有行
                    rows = []
                    for line in lines:
                        if not line.strip():
                            continue
                            
                        row = []
                        last_pos = 0
                        for pos in space_positions:
                            if pos > last_pos:
                                cell = line[last_pos:pos].strip()
                                if cell:
                                    row.append(cell)
                                last_pos = pos
                        
                        # 添加最后一列
                        if last_pos < len(line):
                            cell = line[last_pos:].strip()
                            if cell:
                                row.append(cell)
                                
                        if row:
                            rows.append(row)
                    
                    # 确保所有行有相同的列数
                    if rows:
                        max_cols = max(len(row) for row in rows)
                        for i, row in enumerate(rows):
                            if len(row) < max_cols:
                                rows[i] = row + [''] * (max_cols - len(row))
                        
                        # 第一行作为表头
                        headers = rows[0] if len(rows) > 1 else [f"列{i+1}" for i in range(max_cols)]
                        data = rows[1:] if len(rows) > 1 else rows
                        
                        df = pd.DataFrame(data, columns=headers)
                        if not df.empty:
                            return df
        except Exception:
            pass
        
        # 方法4: 尝试使用制表符分隔
        try:
            if '\t' in text:
                df = pd.read_csv(io.StringIO(text), sep='\t', skipinitialspace=True)
                if not df.empty and len(df.columns) > 1:
                    return df
        except Exception:
            pass
        
        # 方法5: 尝试其他常见分隔符
        for sep in [',', ';']:
            try:
                if sep in text:
                    df = pd.read_csv(io.StringIO(text), sep=sep, skipinitialspace=True)
                    if not df.empty and len(df.columns) > 1:
                        return df
            except Exception:
                continue
        
        # 如果所有方法都失败，返回None
        return None
    except Exception as e:
        logger.error(f"转换表格时出错: {str(e)}")
        return None

def get_chat_response(
    query: str,
    history: List[Tuple[str, str]],
    use_rag: bool = False,
    top_k: int = 3,
    strategy: str = "事实检索"
) -> Tuple[str, List[Tuple[str, str]], str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """获取聊天回复"""
    global client, rag_system, chat_history, table_extractor
    
    try:
        start_time = time.time()
        context = ""
        contexts = []
        source_df = None
        table_df = None
        searched_table_df = None
        
        # 构建消息历史
        messages = [{"role": "system", "content": "你是一个专业、友好的AI助手，擅长财务数据分析。你应该以精确的方式回答关于财务数据的问题，直接引用相关数字。"}]
        
        # 添加聊天历史
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # 如果启用RAG并且有查询内容
        if use_rag and query and rag_system:
            # 将中文策略名转换为英文
            strategy_map = {
                "事实检索": "factual",
                "分析检索": "analytical",
                "观点检索": "opinion",
                "上下文检索": "contextual"
            }
            strategy_eng = strategy_map.get(strategy, strategy)
            
            # 确保切换检索策略
            if rag_system.document_retriever.strategy.__class__.__name__.lower().replace('retrievalstrategy', '') != strategy_eng:
                rag_system.document_retriever.change_strategy(strategy_eng)
            
            # 增强查询，使其更适合财务数据检索
            enhanced_query = query
            if "收入" in query or "revenue" in query.lower() or "income" in query.lower():
                enhanced_query = f"{query} revenue income 收入 总收入 营业收入 营收"
            elif "利润" in query or "profit" in query.lower():
                enhanced_query = f"{query} profit net income 利润 净利润"
            elif "资产" in query or "asset" in query.lower():
                enhanced_query = f"{query} assets 资产 总资产"
                
            # 检索相关文档
            contexts = rag_system.document_retriever.retrieve(enhanced_query, top_k=top_k)
            
            # 尝试搜索相关表格
            try:
                if table_extractor:
                    # 检测是否包含表格相关查询
                    table_keywords = ["表格", "table", "数据表", "报表", "财务表"]
                    is_table_query = any(keyword in query for keyword in table_keywords)
                    
                    # 如果是表格查询，或者包含财务关键词
                    if is_table_query or "收入" in query or "利润" in query or "资产" in query:
                        tables = table_extractor.search_tables(query, top_k=2)
                        if tables:
                            # 获取第一个表格的数据
                            first_table = tables[0]
                            searched_table_df = table_extractor.get_table_by_id(first_table['id'])
                            
                            # 添加表格信息到上下文
                            table_context = f"\n\n查询找到了相关表格:\n{format_table_for_display(first_table, searched_table_df)}"
                            contexts.append(table_context)
            except Exception as e:
                print(f"搜索表格时出错: {str(e)}")
            
            if contexts:
                # 创建更详细的上下文显示
                context_parts = []
                processed_contexts = []
                
                # 检测是否包含表格
                has_table = False
                table_text = ""
                
                for i, ctx in enumerate(contexts):
                    # 查找文档源
                    source = "未知"
                    for doc in rag_system.document_retriever.documents_with_embeddings:
                        if doc[0] == ctx:
                            source = doc[2]
                            break
                    
                    # 检测并处理表格
                    contains_table, formatted_text = detect_and_format_table(ctx)
                    if contains_table and not has_table:
                        has_table = True
                        table_text = formatted_text
                        
                        # 尝试转换为DataFrame
                        table_df = convert_text_to_dataframe(formatted_text)
                        if table_df is not None and not table_df.empty:
                            logger.info(f"成功从上下文中提取表格，列数: {len(table_df.columns)}, 行数: {len(table_df)}")
                    
                    # 预处理上下文，识别财务数据
                    processed_ctx = preprocess_financial_context(ctx)
                    processed_contexts.append(processed_ctx)
                    
                    context_parts.append(f"相关信息 {i+1} (来源: {source}):\n{ctx}")
                
                # 为显示准备的上下文
                context = "\n\n".join(context_parts)
                
                # 添加表格信息到处理后的上下文
                table_prompt = ""
                if has_table and table_df is not None and not table_df.empty:
                    # 将DataFrame转为更易读的格式
                    table_str = table_df.to_string(index=False)
                    table_prompt = f"\n\n文档中包含表格数据，请特别关注这些表格数据：\n{table_str}"
                
                # 如果找到了相关表格，添加到提示
                if searched_table_df is not None:
                    # 将表格转换为CSV文本
                    csv_buffer = io.StringIO()
                    searched_table_df.to_csv(csv_buffer, index=False)
                    table_csv = csv_buffer.getvalue()
                    
                    table_prompt += f"\n\n此外，找到了与查询直接相关的表格数据：\n```\n{table_csv}\n```\n请优先使用这个表格回答问题。"
                
                # 为LLM准备的上下文（带有标记的财务数据）
                processed_context = "\n\n".join([
                    f"相关信息 {i+1}:\n{ctx}" for i, ctx in enumerate(processed_contexts)
                ]) + table_prompt
                
                # 为GPT提供更清晰的财务数据提取指示
                messages.append({
                    "role": "system", 
                    "content": f"""请作为一位财务分析师，基于以下财报信息回答用户的问题。

财务专业指引：
1. 首先识别所有提到的财务数据，包括收入、利润、资产等具体数字
2. 区分不同会计年度的数据，确保引用正确的年份数据
3. 准确引用原始数字，不进行非必要转换
4. 明确标识数据单位（如百万、亿等）
5. 直接回答问题，提供准确的数字
6. 如果找到相关信息但不完全匹配用户问题，也请提供并说明差异
7. 如果上下文中有表格数据，一定要特别关注并优先使用表格中的数据回答问题

当前问题涉及财务数据检索，请务必在上下文中寻找具体的数字和金额。
上下文中，【金额】【百分比】【年份】等标记表示已识别的关键财务数据，请特别注意这些内容。

相关财报信息:
{processed_context}"""
                })
                
                # 生成数据统计
                if rag_system.document_retriever.documents_with_embeddings:
                    sources = {}
                    for _, _, source in rag_system.document_retriever.documents_with_embeddings:
                        sources[source] = sources.get(source, 0) + 1
                    source_df = pd.DataFrame({
                        "来源": list(sources.keys()),
                        "文档块数量": list(sources.values())
                    })
        
        # 添加当前用户问题 - 为财务查询添加明确指令
        if "收入" in query or "利润" in query or "资产" in query:
            query_with_instruction = f"请提取并直接回答这个财务问题，以具体数字作为答案: {query}"
        else:
            query_with_instruction = query
            
        messages.append({"role": "user", "content": query_with_instruction})
        
        # 调用OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,  # 降低温度使回答更加精确
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        process_time = f"处理时间: {time.time() - start_time:.2f} 秒"
        
        # 如果没有找到相关信息，给出更友好的提示
        if use_rag and not contexts:
            answer = f"抱歉，我没有找到关于这个问题的相关信息。当前使用的检索策略是{strategy}，您可以尝试：\n1. 使用其他检索策略\n2. 增加返回文档数量\n3. 调整问题描述，使用更准确的关键词"
        
        # 格式化表格显示
        if table_df is not None:
            table_df = format_table_display(table_df)
            
        if searched_table_df is not None:
            searched_table_df = format_table_display(searched_table_df)
        
        # 更新聊天历史
        history.append((query, answer))
        chat_history = history
        
        return answer, history, context, source_df, table_df, searched_table_df
        
    except Exception as e:
        error_msg = f"生成回答时出错: {str(e)}"
        return error_msg, history, "", None, None, None

def clear_history():
    """清除对话历史"""
    global chat_history
    chat_history = []
    return [], ""

def create_ui():
    """创建Gradio界面"""
    with gr.Blocks(title="财报信息问答系统") as interface:
        gr.Markdown("# 财报信息问答系统")
        gr.Markdown("基于GPT-4的对话系统，支持财报数据检索增强生成")
        
        with gr.Tabs() as tabs:
            # 对话界面
            with gr.TabItem("对话问答"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            process_btn = gr.Button("处理新财报文件")
                            reprocess_btn = gr.Button("重新处理所有财报")
                        process_output = gr.Textbox(label="处理结果")
                        
                        use_rag = gr.Checkbox(
                            label="启用知识库检索",
                            value=False
                        )
                        
                        with gr.Group(visible=False) as rag_settings:
                            gr.Markdown("### 检索设置")
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="返回文档数量"
                            )
                            
                            strategy = gr.Dropdown(
                                choices=["事实检索", "分析检索", "观点检索", "上下文检索"],
                                value="事实检索",
                                label="检索策略"
                            )
                        
                        # 动态显示/隐藏检索设置
                        use_rag.change(
                            lambda x: gr.Group(visible=x),
                            inputs=[use_rag],
                            outputs=[rag_settings]
                        )
                    
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="对话历史",
                            height=500
                        )
                        
                        with gr.Row():
                            query_input = gr.Textbox(
                                label="请输入您的问题",
                                placeholder="例如：你好，你是谁？"
                            )
                            query_btn = gr.Button("发送")
                            clear_btn = gr.Button("清除历史")
                        
                        contexts_output = gr.Textbox(
                            label="相关上下文",
                            max_lines=15,
                            visible=False
                        )
                        
                        with gr.Row(visible=True) as table_outputs:
                            table_output = gr.Dataframe(
                                label="检索到的表格数据",
                                visible=True
                            )
                            
                            searched_table_output = gr.Dataframe(
                                label="相关表格数据",
                                visible=True
                            )
                        
                        process_time_output = gr.Textbox(
                            label="处理信息",
                            visible=False
                        )
                        
                        gr.Markdown("### 数据统计")
                        stats_output = gr.Dataframe(
                            headers=["来源", "文档块数量"],
                            label="文档来源分布",
                            visible=False
                        )
            
            # 表格提取界面
            with gr.TabItem("表格提取"):
                with gr.Row():
                    with gr.Column(scale=1):
                        extract_btn = gr.Button("提取财报表格")
                        extract_output = gr.Textbox(
                            label="提取结果", 
                            max_lines=15
                        )
                        
                        table_search_input = gr.Textbox(
                            label="表格搜索",
                            placeholder="输入关键词搜索表格，例如：收入 2023"
                        )
                        table_search_btn = gr.Button("搜索表格")
                        
                    with gr.Column(scale=2):
                        table_search_result = gr.Textbox(
                            label="搜索结果", 
                            max_lines=10
                        )
                        
                        table_preview = gr.Dataframe(
                            label="表格预览"
                        )
        
        # 事件处理
        process_btn.click(
            process_financial_reports,
            outputs=[process_output]
        )
        
        reprocess_btn.click(
            reprocess_financial_reports,
            outputs=[process_output]
        )
        
        # 表格提取和搜索
        extract_btn.click(
            extract_tables,
            outputs=[extract_output]
        )
        
        table_search_btn.click(
            search_tables,
            inputs=[table_search_input],
            outputs=[table_search_result, table_preview]
        )
        
        # 清除对话历史
        clear_btn.click(
            clear_history,
            outputs=[chatbot, query_input]
        )
        
        # 更新上下文和统计信息的可见性
        use_rag.change(
            lambda x: [gr.Textbox(visible=x), gr.Textbox(visible=x), gr.Dataframe(visible=x)],
            inputs=[use_rag],
            outputs=[contexts_output, process_time_output, stats_output]
        )
        
        # 发送消息处理
        query_btn.click(
            get_chat_response,
            inputs=[
                query_input,
                chatbot,
                use_rag,
                top_k,
                strategy
            ],
            outputs=[
                query_input,
                chatbot,
                contexts_output,
                stats_output,
                table_output,
                searched_table_output
            ]
        ).then(
            lambda: "",
            None,
            [query_input]
        )
        
        # 初始化系统
        if not initialize_rag_system():
            gr.Warning("RAG系统初始化失败，请检查配置和依赖项。")
        
        # 检查嵌入向量
        if (not os.path.exists(EMBEDDING_FILE) or 
            not rag_system.document_retriever.documents_with_embeddings):
            gr.Warning("未找到嵌入向量文件或文件为空。请先处理财报文件。")
    
    return interface

def dataframe_to_markdown(df):
    """
    将DataFrame转换为Markdown表格
    @param df: pandas DataFrame
    @return: markdown字符串
    """
    if df is None or df.empty:
        return "无有效表格数据"
        
    # 获取列名
    headers = df.columns.tolist()
    
    # 创建表头行
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    
    # 创建分隔行
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    # 创建数据行
    data_rows = []
    for _, row in df.iterrows():
        data_row = "| " + " | ".join(str(val) for val in row.tolist()) + " |"
        data_rows.append(data_row)
    
    # 组合所有行
    markdown_table = "\n".join([header_row, separator_row] + data_rows)
    
    return markdown_table

def format_table_display(df):
    """
    格式化表格以优化显示效果
    @param df: pandas DataFrame
    @return: 格式化后的DataFrame
    """
    if df is None or df.empty:
        return None
    
    # 格式化数值列
    for col in df.columns:
        # 检查是否为数值列
        if pd.api.types.is_numeric_dtype(df[col]):
            # 格式化数字，添加千位分隔符
            df[col] = df[col].apply(lambda x: f"{x:,}" if pd.notnull(x) else "")
    
    # 处理可能的重复列名
    if df.columns.duplicated().any():
        df.columns = [f"{col}_{i}" if i > 0 else col 
                      for i, col in enumerate(pd.Series(df.columns).value_counts().index)]
    
    return df

if __name__ == "__main__":
    asyncio.set_event_loop(asyncio.new_event_loop())
    interface = create_ui()
    interface.launch(inbrowser=True)