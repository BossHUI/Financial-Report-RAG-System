import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置本地模型目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
EMBEDDING_MODEL_DIR = os.path.join(MODEL_DIR, "embedding", "all-MiniLM-L6-v2")
GENERATION_MODEL_DIR = os.path.join(MODEL_DIR, "generation", "opt-125m")

try:
    # 创建目录
    os.makedirs(EMBEDDING_MODEL_DIR, exist_ok=True)
    os.makedirs(GENERATION_MODEL_DIR, exist_ok=True)

    logger.info("开始下载模型...")

    # 下载并保存嵌入模型
    logger.info("下载嵌入模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save(EMBEDDING_MODEL_DIR)
    logger.info(f"嵌入模型已保存到 {EMBEDDING_MODEL_DIR}")

    # 下载并保存生成模型
    # logger.info("下载生成模型...")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    # tokenizer.save_pretrained(GENERATION_MODEL_DIR)
    # model.save_pretrained(GENERATION_MODEL_DIR)
    # logger.info(f"生成模型已保存到 {GENERATION_MODEL_DIR}")

    logger.info("所有模型下载完成!")
except Exception as e:
    logger.error(f"下载模型时出错: {str(e)}")
    # 提供更详细的错误信息
    import traceback
    logger.error(traceback.format_exc())