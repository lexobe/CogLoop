"""
MAM（Memory Anchor Mechanism）权重更新实现
"""
import numpy as np
from typing import List, Tuple

class MAMWeightUpdater:
    """MAM 权重更新器
    
    实现基于间隔重复效应的记忆权重更新机制
    """
    
    def __init__(self, beta: float = 0.8, gamma: float = 1.0, b: float = 0.1):
        """初始化 MAM 权重更新器
        
        Args:
            beta: 旧记忆残留因子 (0.7 ~ 0.99)
            gamma: 新调用强化增益系数 (0.5 ~ 5)
            b: 时间敏感系数 (0.01 ~ 0.2)
        """
        self.beta = beta
        self.gamma = gamma
        self.b = b
        
    def update_weight(self, current_weight: float, time_delta: float) -> float:
        """更新单个认元的权重
        
        Args:
            current_weight: 当前权重
            time_delta: 时间间隔
            
        Returns:
            更新后的权重
        """
        decay_factor = np.exp(-self.b * time_delta)
        new_weight = decay_factor * (current_weight * self.beta + self.gamma * time_delta)
        return new_weight
        
    def update_weights_batch(self, weights: List[float], time_deltas: List[float]) -> List[float]:
        """批量更新认元权重
        
        Args:
            weights: 当前权重列表
            time_deltas: 时间间隔列表
            
        Returns:
            更新后的权重列表
        """
        return [self.update_weight(w, dt) for w, dt in zip(weights, time_deltas)]
        
    def get_optimal_interval(self) -> float:
        """获取最优时间间隔
        
        根据当前参数计算最优的时间间隔
        
        Returns:
            最优时间间隔
        """
        return 1.0 / self.b 