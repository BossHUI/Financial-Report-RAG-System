import os
import logging
import pickle
import re
import pandas as pd
import numpy as np
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime
import time
import json
import io
from pathlib import Path
from PyPDF2 import PdfReader
import torch

from src.config import (
    TABLES_DIR, 
    TABLE_INDEX_FILE, 
    DOCLING_PATH, 
    LOCAL_LLAMA_PATH, 
    USE_LOCAL_LLAMA,
    DEVICES
)

logger = logging.getLogger(__name__)

class TableExtractor:
    """
    @class TableExtractor
    @description 表格提取器，用于从PDF文档中提取表格
    """
    def __init__(self):
        """初始化表格提取器"""
        self.tables_index = self._load_tables_index()
        self.docling_model = None
        self.llm = None
        
        try:
            # 初始化docling
            if os.path.exists(DOCLING_PATH):
                logger.info(f"加载docling模型: {DOCLING_PATH}")
                self._init_docling()
            else:
                logger.warning(f"Docling模型路径不存在: {DOCLING_PATH}")
                
            # 初始化本地llama模型(如果启用)
            if USE_LOCAL_LLAMA and os.path.exists(LOCAL_LLAMA_PATH):
                logger.info(f"加载本地LLM模型: {LOCAL_LLAMA_PATH}")
                self._init_local_llm()
            else:
                if USE_LOCAL_LLAMA:
                    logger.warning(f"本地LLM模型路径不存在: {LOCAL_LLAMA_PATH}")
                logger.info("将使用OpenAI API进行表格处理")
        except Exception as e:
            logger.error(f"初始化表格提取器时出错: {str(e)}")
            
    def _load_tables_index(self) -> Dict[str, Any]:
        """
        加载表格索引
        @return: 表格索引字典
        """
        if os.path.exists(TABLE_INDEX_FILE):
            try:
                with open(TABLE_INDEX_FILE, 'rb') as f:
                    tables_index = pickle.load(f)
                logger.info(f"已加载表格索引，包含 {len(tables_index)} 个表格")
                return tables_index
            except Exception as e:
                logger.error(f"加载表格索引失败: {str(e)}")
                return {}
        else:
            logger.info("表格索引文件不存在，将创建新索引")
            return {}
            
    def _save_tables_index(self) -> None:
        """保存表格索引"""
        try:
            with open(TABLE_INDEX_FILE, 'wb') as f:
                pickle.dump(self.tables_index, f)
            logger.info(f"已保存表格索引，包含 {len(self.tables_index)} 个表格")
        except Exception as e:
            logger.error(f"保存表格索引失败: {str(e)}")
            
    def _init_docling(self) -> None:
        """初始化docling模型"""
        try:
            # 导入docling
            import sys
            if DOCLING_PATH not in sys.path:
                sys.path.append(DOCLING_PATH)
                
            try:
                from docling import DocLing, load_docling
                
                # 加载docling模型
                self.docling_model = load_docling()
                logger.info("成功加载Docling模型")
            except ImportError:
                logger.error("导入docling失败，请确保已安装相关依赖")
                logger.info("将尝试使用Docling API或备用方法")
                self.docling_model = None
        except Exception as e:
            logger.error(f"初始化Docling模型失败: {str(e)}")
            self.docling_model = None
            
    def _init_local_llm(self) -> None:
        """初始化本地LLM模型"""
        try:
            # 检查是否支持GPU
            device = DEVICES
            
            from llama_cpp import Llama
            
            # 查找模型文件
            model_files = [f for f in os.listdir(LOCAL_LLAMA_PATH) if f.endswith('.gguf')]
            if not model_files:
                logger.error(f"在 {LOCAL_LLAMA_PATH} 中未找到GGUF模型文件")
                return
                
            model_path = os.path.join(LOCAL_LLAMA_PATH, model_files[0])
            logger.info(f"使用模型: {model_path}")
            
            # 加载模型
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # 上下文长度
                n_gpu_layers=-1 if device == "cuda" else 0  # 使用尽可能多的GPU层
            )
            
            logger.info("成功加载本地LLM模型")
        except ImportError:
            logger.error("导入llama_cpp失败，请确保已安装llama-cpp-python")
            self.llm = None
        except Exception as e:
            logger.error(f"初始化本地LLM失败: {str(e)}")
            self.llm = None
    
    def _generate_with_local_llm(self, prompt: str) -> str:
        """
        使用本地LLM生成响应
        @param prompt: 提示文本
        @return: 生成的响应
        """
        if not self.llm:
            logger.error("本地LLM未初始化")
            return "无法生成响应：本地LLM未初始化"
            
        try:
            # 生成响应
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.1,
                stop=["</answer>", "\n\n"]
            )
            
            # 提取生成的文本
            generated_text = response["choices"][0]["text"].strip()
            return generated_text
        except Exception as e:
            logger.error(f"本地LLM生成失败: {str(e)}")
            return f"生成失败: {str(e)}"
            
    def _generate_with_openai(self, prompt: str) -> str:
        """
        使用OpenAI API生成响应
        @param prompt: 提示文本
        @return: 生成的响应
        """
        try:
            from openai import OpenAI
            import os
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("未设置OPENAI_API_KEY，无法使用OpenAI API")
                return "无法生成响应：未设置OpenAI API密钥"
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一个专业的表格解析助手，擅长将文本转换为结构化的表格数据。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API请求失败: {str(e)}")
            return f"生成失败: {str(e)}"
            
    def _parse_table_with_llm(self, table_text: str) -> pd.DataFrame:
        """
        使用LLM解析表格文本
        @param table_text: 表格文本
        @return: 解析后的DataFrame
        """
        prompt = f"""请将以下文本解析为结构化表格。返回一个包含表头和数据的CSV格式结果。
        如果文本中包含多个表格，请只解析最主要的那个。
        请尽量保持原始数据的完整性，包括所有数字和单位。
        
        表格文本:
        ```
        {table_text}
        ```
        
        请输出解析后的CSV表格:"""
        
        # 选择生成方式
        if USE_LOCAL_LLAMA and self.llm:
            response = self._generate_with_local_llm(prompt)
        else:
            response = self._generate_with_openai(prompt)
            
        try:
            # 提取CSV部分
            csv_match = re.search(r'```(?:csv)?\s*([\s\S]+?)```', response)
            if csv_match:
                csv_text = csv_match.group(1).strip()
            else:
                csv_text = response.strip()
                
            # 解析CSV前进行清理和规范化
            cleaned_csv = self._clean_csv_text(csv_text)
            
            # 使用更灵活的参数进行解析
            df = pd.read_csv(
                io.StringIO(cleaned_csv),
                on_bad_lines='skip',    # 跳过格式有问题的行
                delimiter=None,         # 自动检测分隔符
                engine='python'         # 使用更灵活的解析引擎
            )
            return df
        except Exception as e:
            logger.error(f"解析LLM生成的表格失败: {str(e)}")
            
            # 尝试直接从文本创建表格
            try:
                # 分割行
                lines = table_text.strip().split('\n')
                
                # 分割列
                rows = []
                for line in lines:
                    # 尝试不同的分隔符
                    for sep in ['|', '\t', '  ']:
                        columns = [col.strip() for col in line.split(sep) if col.strip()]
                        if len(columns) > 1:
                            rows.append(columns)
                            break
                
                if not rows:
                    raise ValueError("无法识别表格结构")
                    
                # 计算列数
                max_cols = max(len(row) for row in rows)
                
                # 规范化行，确保每行列数相同
                normalized_rows = []
                for row in rows:
                    if len(row) < max_cols:
                        row = row + [''] * (max_cols - len(row))
                    normalized_rows.append(row)
                
                # 创建DataFrame
                df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0])
                return df
            except Exception as e2:
                logger.error(f"备用表格解析方法也失败: {str(e2)}")
                # 返回空DataFrame
                return pd.DataFrame()
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """从PDF文件中提取表格"""
        # 检查文件是否已处理
        file_hash = self._get_file_hash(pdf_path)
        if file_hash in self.tables_index:
            logger.info(f"文件 {pdf_path} 已处理，直接返回缓存的表格")
            return self.tables_index[file_hash]['tables']
        
        extracted_tables = []
        
        try:
            # 1. 首先尝试使用pandas直接提取 - 最简单可靠的方法
            try:
                tables = self._extract_tables_with_pandas_direct(pdf_path)
                if tables:
                    extracted_tables.extend(tables)
                    logger.info(f"使用pandas从 {pdf_path} 提取了 {len(tables)} 个表格")
            except Exception as e:
                logger.debug(f"使用pandas提取表格失败: {str(e)}")
            
            # 2. 如果pandas方法失败，尝试PyMuPDF
            if not extracted_tables:
                try:
                    tables = self._extract_tables_with_pymupdf(pdf_path)
                    if tables:
                        extracted_tables.extend(tables)
                        logger.info(f"使用PyMuPDF从 {pdf_path} 提取了 {len(tables)} 个表格")
                except Exception as e:
                    logger.debug(f"使用PyMuPDF提取表格失败: {str(e)}")
            
            # 3. 如果前两种方法都失败，尝试OCR（对于扫描版PDF）
            if not extracted_tables:
                try:
                    tables = self._extract_tables_with_ocr(pdf_path)
                    if tables:
                        extracted_tables.extend(tables)
                        logger.info(f"使用OCR从 {pdf_path} 提取了 {len(tables)} 个表格")
                except Exception as e:
                    logger.debug(f"使用OCR提取表格失败: {str(e)}")
            
            # 4. 使用备用方法或docling作为最后手段
            if not extracted_tables:
                if self.docling_model:
                    try:
                        tables = self._extract_tables_with_docling(pdf_path)
                        if tables:
                            extracted_tables.extend(tables)
                            logger.info(f"使用Docling从 {pdf_path} 提取了 {len(tables)} 个表格")
                    except Exception as e:
                        logger.debug(f"使用Docling提取表格失败: {str(e)}")
                
                if not extracted_tables:
                    tables = self._extract_tables_with_backup(pdf_path)
                    if tables:
                        extracted_tables.extend(tables)
                        logger.info(f"使用备用方法从 {pdf_path} 提取了 {len(tables)} 个表格")
            
            # 更新表格索引
            if extracted_tables:
                self.tables_index[file_hash] = {
                    'file_path': pdf_path,
                    'extraction_time': datetime.now().isoformat(),
                    'tables': extracted_tables
                }
                self._save_tables_index()
            
            if not extracted_tables:
                # 记录未提取到表格的情况，但降低日志级别
                logger.info(f"从 {pdf_path} 未提取到表格")
            
            return extracted_tables
            
        except Exception as e:
            logger.error(f"从 {pdf_path} 提取表格时出错: {str(e)}")
            return []
            
    def _extract_tables_with_docling(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        使用docling提取表格
        @param pdf_path: PDF文件路径
        @return: 表格列表
        """
        if not self.docling_model:
            logger.warning("Docling模型未初始化，无法使用此方法提取表格")
            return []
            
        try:
            from docling import DocLing, load_docling
            
            # 加载PDF文档
            doc = self.docling_model.load(pdf_path)
            
            # 提取表格
            tables = []
            for i, page in enumerate(doc.pages):
                page_tables = page.find_tables()
                if page_tables:
                    logger.info(f"在页面 {i+1} 中发现 {len(page_tables)} 个表格")
                    
                    for j, table in enumerate(page_tables):
                        # 提取表格文本和数据
                        table_id = f"{os.path.basename(pdf_path)}_page{i+1}_table{j+1}"
                        table_text = table.extract_text().strip()
                        
                        # 确保表格文本不为空
                        if not table_text:
                            continue
                            
                        # 创建表格条目
                        table_entry = {
                            'id': table_id,
                            'page': i + 1,
                            'text': table_text,
                            'source': pdf_path
                        }
                        
                        # 使用LLM解析表格
                        df = self._parse_table_with_llm(table_text)
                        if not df.empty:
                            # 保存表格数据
                            table_file = os.path.join(TABLES_DIR, f"{table_id}.pkl")
                            with open(table_file, 'wb') as f:
                                pickle.dump(df, f)
                                
                            # 添加表格数据信息
                            table_entry['table_file'] = table_file
                            table_entry['columns'] = df.columns.tolist()
                            table_entry['rows'] = len(df)
                            
                            # 添加CSV格式
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            table_entry['csv'] = csv_buffer.getvalue()
                        
                        tables.append(table_entry)
            
            logger.info(f"使用Docling从 {pdf_path} 提取了 {len(tables)} 个表格")
            return tables
            
        except Exception as e:
            logger.error(f"使用Docling提取表格时出错: {str(e)}")
            return []
    
    def _extract_tables_with_backup(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        使用备用方法提取表格
        @param pdf_path: PDF文件路径
        @return: 表格列表
        """
        try:
            # 使用PyPDF2读取PDF
            reader = PdfReader(pdf_path)
            
            # 提取页面文本并识别表格
            tables = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if not page_text:
                        continue
                        
                    # 检测表格
                    table_sections = self._detect_tables_in_text(page_text)
                    
                    if table_sections:
                        logger.info(f"在页面 {i+1} 中检测到 {len(table_sections)} 个可能的表格")
                        
                        for j, table_text in enumerate(table_sections):
                            table_id = f"{os.path.basename(pdf_path)}_page{i+1}_table{j+1}"
                            
                            # 创建表格条目
                            table_entry = {
                                'id': table_id,
                                'page': i + 1,
                                'text': table_text,
                                'source': pdf_path
                            }
                            
                            # 使用LLM解析表格
                            df = self._parse_table_with_llm(table_text)
                            if not df.empty:
                                # 保存表格数据
                                table_file = os.path.join(TABLES_DIR, f"{table_id}.pkl")
                                with open(table_file, 'wb') as f:
                                    pickle.dump(df, f)
                                    
                                # 添加表格数据信息
                                table_entry['table_file'] = table_file
                                table_entry['columns'] = df.columns.tolist()
                                table_entry['rows'] = len(df)
                                
                                # 添加CSV格式
                                csv_buffer = io.StringIO()
                                df.to_csv(csv_buffer, index=False)
                                table_entry['csv'] = csv_buffer.getvalue()
                            
                            tables.append(table_entry)
                            
                except Exception as e:
                    logger.error(f"处理页面 {i+1} 时出错: {str(e)}")
            
            logger.info(f"使用备用方法从 {pdf_path} 提取了 {len(tables)} 个表格")
            return tables
            
        except Exception as e:
            logger.error(f"使用备用方法提取表格时出错: {str(e)}")
            return []
    
    def _detect_tables_in_text(self, text: str) -> List[str]:
        """在文本中检测表格"""
        table_sections = []
        lines = text.split('\n')
        
        # 增强表格检测特征
        table_patterns = [
            r'(\|[-+]+\|)+',  # Markdown表格分隔符
            r'(\+[-+]+\+)+',  # ASCII表格分隔符
            r'((\d+[.,]?\d*\s*){3,})',  # 连续多个数字
            r'([\w\s]+\s*\d+[.,]?\d*\s*){3,}',  # 标题+数字模式
            r'\b([\w\s]+)\s+(\d{4})\s+(\d{4})',  # 项目名+年份+年份模式(常见于财报)
            r'(\$\s*\d+[.,]?\d*\s*){2,}',  # 多个金额
            r'(人民币\s*\d+[.,]?\d*\s*){2,}',  # 中文金额
            r'(¥\s*\d+[.,]?\d*\s*){2,}'  # 人民币符号+金额
        ]
        
        # 扩展财务表格关键词
        financial_keywords = [
            r'income statement', r'balance sheet', r'cash flow', r'statement of', 
            r'revenue', r'profit', r'margin', r'total', r'fiscal year', r'financial',
            r'利润表', r'资产负债表', r'现金流量表', r'收入', r'利润', r'毛利率', r'总计', 
            r'财年', r'财务报表', r'财务状况', r'年报', r'季报', r'中期报告'
        ]
        
        # 使用更严格的表格模式检测
        consecutive_numeric_lines = 0
        current_table = []
        in_table = False
        
        for i, line in enumerate(lines):
            is_table_line = False
            
            # 检查是否包含数字比例较高(财务表格特征)
            total_chars = len(line.strip())
            if total_chars > 0:
                digit_chars = sum(1 for c in line if c.isdigit())
                digit_ratio = digit_chars / total_chars
                if digit_ratio > 0.2 and digit_ratio < 0.7 and digit_chars > 5:
                    is_table_line = True
                    
            # 检查表格模式
            if not is_table_line:
                for pattern in table_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        is_table_line = True
                        break
            
            # 检查财务关键词+数字的组合
            if not is_table_line and i > 0 and i < len(lines) - 1:
                for keyword in financial_keywords:
                    if re.search(keyword, line, re.IGNORECASE):
                        # 前后检查是否有数字行
                        prev_has_numbers = bool(re.search(r'\d+', lines[i-1]))
                        next_has_numbers = bool(re.search(r'\d+', lines[i+1]))
                        if prev_has_numbers or next_has_numbers:
                            is_table_line = True
                            break
            
            # 处理表格状态
            if is_table_line:
                consecutive_numeric_lines += 1
                if not in_table and consecutive_numeric_lines >= 2:
                    in_table = True
                    # 如果有前导行，也加入表格
                    if i > 0 and consecutive_numeric_lines == 2:
                        current_table.append(lines[i-1])
                    current_table.append(line)
                elif in_table:
                    current_table.append(line)
            elif in_table:
                # 检查是否是表格延续行
                if re.search(r'\d+', line) or self._check_alignment(line, current_table[-1]):
                    current_table.append(line)
                    consecutive_numeric_lines = 0  # 重置计数
                else:
                    # 表格结束
                    consecutive_numeric_lines = 0
                    if len(current_table) >= 3:  # 至少需要3行才可能是表格
                        table_sections.append('\n'.join(current_table))
                    in_table = False
                    current_table = []
            else:
                consecutive_numeric_lines = 0
        
        # 处理最后一个表格
        if in_table and len(current_table) >= 3:
            table_sections.append('\n'.join(current_table))
        
        # 确保表格文本足够有效
        filtered_sections = []
        for section in table_sections:
            # 检查行数和平均行长度
            lines = section.strip().split('\n')
            if len(lines) >= 3:
                avg_line_len = sum(len(line) for line in lines) / len(lines)
                if avg_line_len > 20:  # 平均行长度应该足够长
                    filtered_sections.append(section)
        
        return filtered_sections
    
    def _check_alignment(self, line1: str, line2: str) -> bool:
        """
        检查两行是否对齐（可能是表格的一部分）
        @param line1: 第一行
        @param line2: 第二行
        @return: 是否对齐
        """
        # 找出所有空格位置
        spaces1 = [m.start() for m in re.finditer(r'\s{2,}', line1)]
        spaces2 = [m.start() for m in re.finditer(r'\s{2,}', line2)]
        
        # 检查对齐程度
        if not spaces1 or not spaces2:
            return False
            
        # 计算相似度
        matches = 0
        for s1 in spaces1:
            for s2 in spaces2:
                if abs(s1 - s2) <= 2:  # 允许2个字符的误差
                    matches += 1
                    break
                    
        # 如果至少有2个对齐点，认为是对齐的
        return matches >= 2
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        计算文件哈希值
        @param file_path: 文件路径
        @return: 哈希值
        """
        try:
            with open(file_path, 'rb') as f:
                # 只读取前10MB计算哈希，避免大文件处理慢
                content = f.read(10 * 1024 * 1024)
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希值时出错: {str(e)}")
            # 使用文件路径和大小作为备用
            file_stat = os.stat(file_path)
            return hashlib.md5(f"{file_path}_{file_stat.st_size}".encode()).hexdigest()
    
    def extract_tables_from_folder(self, folder_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        从文件夹中提取所有PDF的表格
        @param folder_path: 文件夹路径
        @return: 表格字典，键为文件路径，值为表格列表
        """
        results = {}
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    logger.info(f"处理文件: {file_path}")
                    
                    tables = self.extract_tables_from_pdf(file_path)
                    if tables:
                        results[file_path] = tables
                        logger.info(f"从 {file_path} 提取了 {len(tables)} 个表格")
                    else:
                        logger.warning(f"从 {file_path} 未提取到表格")
        
        return results
    
    def get_table_by_id(self, table_id: str) -> Optional[pd.DataFrame]:
        """
        根据表格ID获取表格数据
        @param table_id: 表格ID
        @return: 表格DataFrame
        """
        # 查找表格文件
        table_file = os.path.join(TABLES_DIR, f"{table_id}.pkl")
        
        if os.path.exists(table_file):
            try:
                with open(table_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"加载表格数据失败: {str(e)}")
                return None
        else:
            # 在索引中查找
            for file_hash, file_info in self.tables_index.items():
                for table in file_info['tables']:
                    if table['id'] == table_id:
                        if 'table_file' in table and os.path.exists(table['table_file']):
                            try:
                                with open(table['table_file'], 'rb') as f:
                                    return pickle.load(f)
                            except Exception as e:
                                logger.error(f"加载表格数据失败: {str(e)}")
                        elif 'csv' in table:
                            try:
                                return pd.read_csv(io.StringIO(table['csv']))
                            except Exception as e:
                                logger.error(f"从CSV加载表格数据失败: {str(e)}")
        
        return None
    
    def search_tables(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索相关表格
        @param query: 查询文本
        @param top_k: 返回的表格数量
        @return: 相关表格列表
        """
        if not self.tables_index:
            logger.warning("表格索引为空，无法搜索")
            return []
            
        # 关键词搜索
        keywords = query.lower().split()
        results = []
        
        for file_hash, file_info in self.tables_index.items():
            for table in file_info['tables']:
                # 计算相关性得分
                score = 0
                text = table.get('text', '').lower()
                
                # 检查每个关键词
                for keyword in keywords:
                    if keyword in text:
                        score += 1
                
                # 检查表格列名
                if 'columns' in table:
                    columns_text = ' '.join(table['columns']).lower()
                    for keyword in keywords:
                        if keyword in columns_text:
                            score += 2  # 列名匹配权重更高
                
                if score > 0:
                    # 添加得分到表格信息
                    table_copy = table.copy()
                    table_copy['score'] = score
                    results.append(table_copy)
        
        # 按得分排序并返回前top_k个
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_all_tables(self) -> List[Dict[str, Any]]:
        """
        获取所有表格
        @return: 所有表格的列表
        """
        results = []
        
        for file_hash, file_info in self.tables_index.items():
            for table in file_info['tables']:
                results.append(table)
                
        return results 
    
    def _extract_tables_with_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """使用PyMuPDF从PDF中提取表格"""
        try:
            import fitz  # PyMuPDF
            
            tables = []
            doc = fitz.open(pdf_path)
            
            # 遍历页面
            for page_idx, page in enumerate(doc):
                try:
                    # 提取文本块
                    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
                    
                    # 识别表格区域
                    table_regions = []
                    for b in blocks:
                        try:
                            if "lines" in b and len(b["lines"]) > 2:  # 可能的表格块
                                # 检查是否有多列对齐
                                x_positions = {}
                                
                                for line in b["lines"]:
                                    for span in line.get("spans", []):
                                        # 安全获取x0值
                                        x = round(span.get("x0", 0))
                                        if x > 0:  # 确保x值有效
                                            if x not in x_positions:
                                                x_positions[x] = 0
                                            x_positions[x] += 1
                        
                                # 如果有多个对齐点，可能是表格
                                if len([k for k, v in x_positions.items() if v > 2]) >= 2:
                                    table_regions.append(b)
                        except KeyError:
                            continue  # 跳过缺少必要键的块
                                
                    # 处理找到的表格区域
                    for i, region in enumerate(table_regions):
                        try:
                            table_text = ""
                            for line in region.get("lines", []):
                                line_text = " ".join([span.get("text", "") for span in line.get("spans", [])])
                                table_text += line_text + "\n"
                            
                            # 跳过过短的表格文本
                            if len(table_text.strip()) < 50:
                                continue
                            
                            # 尝试将文本转换为DataFrame
                            df = self._text_to_dataframe(table_text)
                            
                            if df is not None and not df.empty and len(df.columns) > 1:
                                # 创建表格ID
                                table_id = f"{os.path.basename(pdf_path)}_page{page_idx+1}_table{i+1}"
                                
                                # 创建表格条目
                                table_entry = {
                                    'id': table_id,
                                    'page': page_idx + 1,
                                    'text': table_text,
                                    'source': pdf_path
                                }
                                
                                # 保存表格数据
                                table_file = os.path.join(TABLES_DIR, f"{table_id}.pkl")
                                with open(table_file, 'wb') as f:
                                    pickle.dump(df, f)
                                
                                # 添加表格数据信息
                                table_entry['table_file'] = table_file
                                table_entry['columns'] = df.columns.tolist()
                                table_entry['rows'] = len(df)
                                
                                # 添加CSV格式
                                csv_buffer = io.StringIO()
                                df.to_csv(csv_buffer, index=False)
                                table_entry['csv'] = csv_buffer.getvalue()
                                
                                tables.append(table_entry)
                        except Exception as e:
                            logger.debug(f"处理表格区域时出错: {str(e)}")
                            continue
                except Exception as e:
                    logger.debug(f"处理页面 {page_idx+1} 时出错: {str(e)}")
                    continue
                
                # 如果PyMuPDF方法没有找到表格，尝试备用方法
                if not tables:
                    try:
                        # 获取页面文本
                        page_text = page.get_text()
                        if page_text:
                            # 使用直接文本分析
                            table_sections = self._detect_tables_in_text(page_text)
                            for j, section in enumerate(table_sections):
                                try:
                                    # 使用pandas处理
                                    df = pd.read_fwf(io.StringIO(section))
                                    if not df.empty and len(df.columns) > 1:
                                        # 处理表格 (与上面类似)
                                        table_id = f"{os.path.basename(pdf_path)}_page{page_idx+1}_table{j+1}"
                                        table_entry = {
                                            'id': table_id,
                                            'page': page_idx + 1,
                                            'text': section,
                                            'source': pdf_path
                                        }
                                        
                                        # 保存并添加表格信息
                                        table_file = os.path.join(TABLES_DIR, f"{table_id}.pkl")
                                        with open(table_file, 'wb') as f:
                                            pickle.dump(df, f)
                                        
                                        # 添加表格数据信息
                                        table_entry['table_file'] = table_file
                                        table_entry['columns'] = df.columns.tolist()
                                        table_entry['rows'] = len(df)
                                        
                                        # 添加CSV格式
                                        csv_buffer = io.StringIO()
                                        df.to_csv(csv_buffer, index=False)
                                        table_entry['csv'] = csv_buffer.getvalue()
                                        
                                        tables.append(table_entry)
                                except Exception as e:
                                    logger.debug(f"备用方法处理表格 {j+1} 出错: {str(e)}")
                                    continue
                    except Exception:
                        pass
            
            doc.close()
            return tables
            
        except ImportError:
            logger.error("未安装PyMuPDF，请使用 pip install pymupdf 安装")
            return []
        except Exception as e:
            logger.error(f"使用PyMuPDF提取表格时出错: {str(e)}")
            return []
        
    def _text_to_dataframe(self, text: str) -> Optional[pd.DataFrame]:
        """
        将表格文本转换为DataFrame
        @param text: 表格文本
        @return: DataFrame或None
        """
        try:
            # 分割行
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            if len(lines) < 2:  # 至少需要标题行和一行数据
                return None
            
            # 识别列分隔点
            x_positions = {}
            space_runs = []
            
            # 分析每一行中的空格位置
            for line in lines:
                # 查找连续空格
                for match in re.finditer(r'\s{2,}', line):
                    start, end = match.span()
                    mid = (start + end) // 2
                    if mid not in x_positions:
                        x_positions[mid] = 0
                    x_positions[mid] += 1
            
            # 选择频率最高的分隔点
            common_positions = sorted([pos for pos, count in x_positions.items() 
                                      if count >= len(lines) * 0.5])
            
            # 使用分隔点分割每行
            rows = []
            for line in lines:
                row = []
                last_pos = 0
                
                # 按照识别的位置分割
                for pos in common_positions:
                    if pos <= len(line):
                        cell = line[last_pos:pos].strip()
                        row.append(cell)
                        last_pos = pos
                
                # 添加最后一个单元格
                if last_pos < len(line):
                    row.append(line[last_pos:].strip())
                    
                rows.append(row)
            
            # 确保所有行有相同的列数
            max_cols = max(len(row) for row in rows)
            for i, row in enumerate(rows):
                if len(row) < max_cols:
                    rows[i] = row + [''] * (max_cols - len(row))
            
            # 创建DataFrame
            if len(rows) >= 1:
                header = rows[0]
                data = rows[1:]
                df = pd.DataFrame(data, columns=header)
                return df
            
            return None
        except Exception as e:
            logger.error(f"将文本转换为DataFrame时出错: {str(e)}")
            return None

    def _clean_csv_text(self, csv_text: str) -> str:
        """
        清理CSV文本，处理格式问题
        @param csv_text: 原始CSV文本
        @return: 清理后的CSV文本
        """
        # 分行处理
        lines = [line.strip() for line in csv_text.strip().split('\n') if line.strip()]
        if not lines:
            return ""
        
        # 检测分隔符
        if ',' in lines[0]:
            sep = ','
        elif ';' in lines[0]:
            sep = ';'
        elif '\t' in lines[0]:
            sep = '\t'
        else:
            # 默认使用逗号
            sep = ','
        
        # 计算每行的字段数
        field_counts = [len(line.split(sep)) for line in lines]
        if not field_counts:
            return csv_text
        
        # 获取出现最多的字段数作为标准
        most_common_count = max(set(field_counts), key=field_counts.count)
        
        # 规范化每一行
        normalized_lines = []
        for line in lines:
            fields = line.split(sep)
            if len(fields) == most_common_count:
                # 行已经是标准字段数
                normalized_lines.append(line)
            elif len(fields) < most_common_count:
                # 字段太少，添加空字段
                fields += [''] * (most_common_count - len(fields))
                normalized_lines.append(sep.join(fields))
            else:
                # 字段太多，合并多余字段
                normalized_fields = fields[:most_common_count-1] 
                # 将剩余字段合并为最后一个字段
                last_field = sep.join(fields[most_common_count-1:])
                normalized_fields.append(last_field.replace(sep, ' '))
                normalized_lines.append(sep.join(normalized_fields))
        
        return '\n'.join(normalized_lines)

    def _extract_tables_with_pandas_direct(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        使用pandas直接提取表格
        @param pdf_path: PDF文件路径
        @return: 表格列表
        """
        try:
            # 安全读取PDF文件
            reader = self._safe_read_pdf(pdf_path)
            
            tables = []
            for i, page in enumerate(reader.pages):
                try:
                    # 安全获取页面文本
                    try:
                        page_text = page.extract_text()
                    except UnicodeDecodeError:
                        # 尝试使用不同编码
                        page_text = page.extract_text().encode('latin-1').decode('utf-8', errors='ignore')
                    
                    if not page_text:
                        continue
                        
                    # 检测表格
                    table_sections = self._detect_tables_in_text(page_text)
                    
                    for j, table_text in enumerate(table_sections):
                        try:
                            # 使用pandas.read_fwf解析固定宽度的表格
                            df = pd.read_fwf(io.StringIO(table_text))
                            
                            # 清理表格
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            # 至少需要2列，才算是有效表格
                            if not df.empty and len(df.columns) > 1:
                                # 创建表格ID
                                table_id = f"{os.path.basename(pdf_path)}_page{i+1}_table{j+1}"
                                
                                # 创建表格条目
                                table_entry = {
                                    'id': table_id,
                                    'page': i + 1,
                                    'text': table_text,
                                    'source': pdf_path
                                }
                                
                                # 保存表格数据
                                table_file = os.path.join(TABLES_DIR, f"{table_id}.pkl")
                                with open(table_file, 'wb') as f:
                                    pickle.dump(df, f)
                                    
                                # 添加表格数据信息
                                table_entry['table_file'] = table_file
                                table_entry['columns'] = df.columns.tolist()
                                table_entry['rows'] = len(df)
                                
                                # 添加CSV格式
                                csv_buffer = io.StringIO()
                                df.to_csv(csv_buffer, index=False)
                                table_entry['csv'] = csv_buffer.getvalue()
                                
                                tables.append(table_entry)
                        except Exception as e:
                            logger.warning(f"处理表格 {j+1} 时出错: {str(e)}")
                            continue
                except Exception as e:
                    logger.warning(f"处理页面 {i+1} 时出错: {str(e)}")
                    continue
            
            return tables
        except Exception as e:
            logger.error(f"提取表格出错: {str(e)}")
            return []

    def _safe_read_pdf(self, pdf_path: str) -> PdfReader:
        """
        安全读取PDF文件，处理编码问题
        @param pdf_path: PDF文件路径
        @return: PdfReader对象
        """
        try:
            # 尝试直接读取
            return PdfReader(pdf_path)
        except Exception as e:
            if "codec can't decode" in str(e):
                # 尝试二进制模式读取
                with open(pdf_path, 'rb') as f:
                    return PdfReader(f)
            else:
                raise

    def _extract_tables_with_ocr(self, pdf_path: str) -> List[Dict[str, Any]]:
        """使用OCR提取表格(针对扫描版PDF)"""
        try:
            import cv2
            import pytesseract
            import pdf2image
            from pdf2image import convert_from_path
            
            # 设置pytesseract路径(Windows需要)
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            tables = []
            
            # 将PDF转换为图像
            images = convert_from_path(pdf_path)
            
            for i, image in enumerate(images):
                try:
                    # 转换为OpenCV格式
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # 灰度化
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # 二值化
                    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                    
                    # 查找轮廓
                    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # 筛选可能的表格轮廓
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 10000:  # 筛选大的轮廓
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # 提取表格区域
                            roi = gray[y:y+h, x:x+w]
                            
                            # 用OCR识别文本
                            table_text = pytesseract.image_to_string(roi)
                            
                            # 检查是否有表格特征
                            if len(table_text.strip().split('\n')) >= 3:
                                # 尝试解析为DataFrame
                                df = self._text_to_dataframe(table_text)
                                
                                if df is not None and not df.empty and len(df.columns) > 1:
                                    # 创建表格条目
                                    table_id = f"{os.path.basename(pdf_path)}_page{i+1}_ocr_table{len(tables)+1}"
                                    
                                    table_entry = {
                                        'id': table_id,
                                        'page': i + 1,
                                        'text': table_text,
                                        'source': pdf_path
                                    }
                                    
                                    # 保存表格数据
                                    table_file = os.path.join(TABLES_DIR, f"{table_id}.pkl")
                                    with open(table_file, 'wb') as f:
                                        pickle.dump(df, f)
                                    
                                    # 添加表格数据信息
                                    table_entry['table_file'] = table_file
                                    table_entry['columns'] = df.columns.tolist()
                                    table_entry['rows'] = len(df)
                                    
                                    # 添加CSV格式
                                    csv_buffer = io.StringIO()
                                    df.to_csv(csv_buffer, index=False)
                                    table_entry['csv'] = csv_buffer.getvalue()
                                    
                                    tables.append(table_entry)
                
                except Exception as e:
                    logger.debug(f"OCR处理页面 {i+1} 时出错: {str(e)}")
                    continue
                
            return tables
            
        except ImportError:
            logger.warning("OCR依赖不可用，请安装: pip install opencv-python pytesseract pdf2image")
            return []
        except Exception as e:
            logger.error(f"OCR提取表格时出错: {str(e)}")
            return []