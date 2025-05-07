"""
配置加载模块
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# LLM 服务配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Upstash 服务配置
UPSTASH_VECTOR_URL = os.getenv("UPSTASH_VECTOR_URL")
UPSTASH_VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")

def validate_config():
    """验证配置是否完整"""
    required_vars = [
        "OPENAI_API_KEY",
        "UPSTASH_VECTOR_URL",
        "UPSTASH_VECTOR_TOKEN",
        "UPSTASH_REDIS_URL",
        "UPSTASH_REDIS_TOKEN"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"缺少必要的环境变量: {', '.join(missing_vars)}") 