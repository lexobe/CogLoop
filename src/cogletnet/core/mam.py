"""
记忆锚定机制（Memory Anchor Mechanism）的核心计算工具
"""

import math
from typing import Dict, Any, List, Optional
from datetime import datetime

class MAM:
    """记忆锚定机制的核心计算工具类"""
    
    def __init__(
        self,
        beta: float = 0.85,      # 时间衰减系数
        gamma: float = 0.3,      # 访问增强系数
        b: float = 0.05,         # 基础衰减率
        initial_weight: float = 0.5,  # 初始权重
        golden_ratio: float = 0.618,  # 黄金分割比例
    ):
        """
        初始化MAM计算工具
        
        Args:
            beta: 时间衰减系数，控制时间对权重的影响程度
            gamma: 访问增强系数，控制访问对权重的提升程度
            b: 基础衰减率，控制权重的自然衰减速度
            initial_weight: 初始权重值
            golden_ratio: 黄金分割比例，用于记忆激活时的选择比例
        """
        self.beta = beta
        self.gamma = gamma
        self.b = b
        self.initial_weight = initial_weight
        self.golden_ratio = golden_ratio

    def calculate_weight(
        self,
        current_weight: float,
        last_access_time: datetime,
        current_time: datetime,
        access_count: int = 0
    ) -> float:
        """
        计算记忆的权重
        
        Args:
            current_weight: 当前权重
            last_access_time: 上次访问时间
            current_time: 当前时间
            access_count: 访问次数
            
        Returns:
            float: 计算后的新权重
        """
        # 如果是首次创建记忆（current_weight为0）
        if current_weight == 0:
            return self.initial_weight - self.b
            
        # 计算时间差（小时）
        time_diff = (current_time - last_access_time).total_seconds() / 3600
        
        # 计算时间衰减因子
        time_decay = math.exp(-self.beta * time_diff)
        
        # 计算访问增强因子
        access_boost = 1 + (self.gamma * access_count)
        
        # 计算新权重
        new_weight = (current_weight * time_decay * access_boost) - self.b
        
        # 确保权重在[0,1]范围内
        return max(0.0, min(1.0, new_weight))

    def select_activated_memories(
        self,
        memories: List[Dict[str, Any]],
        sort_by: str = 'weight'
    ) -> List[Dict[str, Any]]:
        """
        根据黄金分割比例选择要激活的记忆
        
        Args:
            memories: 记忆列表，每个记忆是一个字典
            sort_by: 排序依据的字段名，默认为'weight'
            
        Returns:
            List[Dict[str, Any]]: 被激活的记忆列表
        """
        if not memories:
            return []
            
        # 按指定字段排序，对于缺失字段使用0作为默认值
        sorted_memories = sorted(
            memories,
            key=lambda x: float(x.get(sort_by, 0)),  # 确保转换为float类型
            reverse=True
        )
        
        # 使用黄金分割比例选择记忆
        # 对于单个记忆，直接返回
        if len(sorted_memories) == 1:
            return sorted_memories
            
        split_index = max(1, int(len(sorted_memories) * self.golden_ratio))
        return sorted_memories[:split_index]
        