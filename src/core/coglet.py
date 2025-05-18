"""
认元（Coglet）基础类实现
"""
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Coglet:
    """认元数据结构
    
    属性:
        id: 认元唯一标识
        embedding: 语义向量
        weight: 权重
        timestamp: 最后更新时间戳
        content: 文本内容
    """
    id: int
    embedding: np.ndarray
    weight: float
    timestamp: float
    content: str 