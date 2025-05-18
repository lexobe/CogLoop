import logging
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

def setup_logger(name="cogletnet"):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称，用于区分不同模块
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 根据模块名称获取对应的环境变量
    env_var = f"LOG_LEVEL_{name}"
    level = os.getenv(env_var, None)
    
    logger = logging.getLogger(name)
    # 避免重复添加 handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # 设置日志级别（如果有env配置才设置，否则保持全局）
        if level:
            logger.setLevel(level.upper())
        
        # 避免日志重复
        logger.propagate = False
        
    
    return logger 