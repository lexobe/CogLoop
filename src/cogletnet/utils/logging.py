"""日志配置模块

提供统一的日志配置和管理功能。
"""

import logging
import os
from dotenv import load_dotenv
from typing import Optional

# 加载 .env 文件
load_dotenv()

def setup_logger(
    name: str = "cogletnet",
    level: Optional[int] = None,
    handler: Optional[logging.Handler] = None,
    propagate: bool = True
) -> logging.Logger:
    """设置库的日志器
    
    Args:
        name: 日志器名称，默认为 'cogletnet'
        level: 日志级别，如果为 None 则使用父日志器的级别
        handler: 自定义的日志处理器，如果为 None 则使用默认的 StreamHandler
        propagate: 是否将日志传播给父日志器
    
    Returns:
        logging.Logger: 配置好的日志器
    """
    logger = logging.getLogger(name)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
        
    # 设置日志级别
    if level is not None:
        logger.setLevel(level)
    
    # 设置处理器
    if handler is None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = propagate
    
    return logger

# 默认日志器
logger = setup_logger() 