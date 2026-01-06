import os
import torch

# 目录配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FINANCIAL_REPORTS_DIR = os.path.join(BASE_DIR, "financial_reports")
MODEL_CACHE_DIR = os.path.join(DATA_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
TABLES_DIR = os.path.join(DATA_DIR, "tables")  # 存储提取的表格

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)  # 确保表格存储目录存在

# 嵌入文件配置
EMBEDDING_FILE = os.path.join(DATA_DIR, "document_embeddings.pkl")
TABLE_INDEX_FILE = os.path.join(DATA_DIR, "table_index.pkl")  # 表格索引文件

# 设备配置
DEVICES = "cuda" if torch.cuda.is_available() else "cpu"

# 检查本地模型目录是否存在
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models/embedding/")
GENERATION_MODEL_PATH = os.path.join(BASE_DIR, "models/generation/")

# 模型配置
if os.path.exists(EMBEDDING_MODEL_PATH) and os.path.exists(GENERATION_MODEL_PATH):
    # 本地模型存在
    MODELS = {
        "embedding": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 8,
            "path": EMBEDDING_MODEL_PATH
        },
       "generation": {
            "name": "gpt-4o",
            "path": None  # 使用OpenAI API
        },

    }
else:
    # 本地模型不存在，使用OpenAI API
    MODELS = {
        "embedding": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 8,
            "path": EMBEDDING_MODEL_PATH
        },
        # "generation": {
        #     "name": "gpt-4o",
        #     "path": None  # 使用OpenAI API
        # },
        "generation": {
            "name": "facebook/opt-125m",
            "path": GENERATION_MODEL_PATH
        }
    }

# 处理配置
BATCH_SIZE = 16
MAX_LENGTH = 4096
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# API配置
MAX_RETRIES = 5  # 增加重试次数
RETRY_DELAY = 5  # 增加基础延迟
REQUEST_TIMEOUT = 120  # 增加API请求超时时间到120秒
API_RATE_LIMIT = 3  # 降低API请求频率，每10秒最多3个请求

# 表格提取配置
TABLE_EXTRACTION_ENABLED = True  # 是否启用表格提取功能
DOCLING_PATH = os.path.join(BASE_DIR, "models/docling/")  # Docling模型路径
LOCAL_LLAMA_PATH = os.path.join(BASE_DIR, "models/llama/")  # 本地Llama模型路径
USE_LOCAL_LLAMA = False  # 是否使用本地Llama模型进行表格处理