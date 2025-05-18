"""
MAM（Memory Anchor Mechanism）的单元测试
"""

import unittest
from datetime import datetime, timedelta
from src.mam import MAM

class TestMAM(unittest.TestCase):
    """MAM类的单元测试"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.mam = MAM(
            beta=0.85,      # 时间衰减系数
            gamma=0.3,      # 访问增强系数
            b=0.05,         # 基础衰减率
            initial_weight=0.5,  # 初始权重
            golden_ratio=0.618,  # 黄金分割比例
        )
        self.current_time = datetime.now()
        
    def test_calculate_weight_initial(self):
        """测试首次创建记忆时的权重计算"""
        # 首次创建记忆，应该返回初始权重减去基础衰减率
        weight = self.mam.calculate_weight(
            current_weight=0.0,
            last_access_time=self.current_time,
            current_time=self.current_time,
            access_count=0
        )
        # 初始权重0.5，减去基础衰减率0.05
        self.assertAlmostEqual(weight, 0.45, places=4)
        
    def test_calculate_weight_time_decay(self):
        """测试时间衰减对权重的影响"""
        # 测试1小时后的权重衰减
        one_hour_later = self.current_time + timedelta(hours=1)
        weight = self.mam.calculate_weight(
            current_weight=0.8,
            last_access_time=self.current_time,
            current_time=one_hour_later,
            access_count=0
        )
        # 计算预期值：0.8 * exp(-0.85 * 1) - 0.05
        expected_weight = 0.8 * 0.4274 - 0.05  # exp(-0.85) ≈ 0.4274
        self.assertAlmostEqual(weight, expected_weight, places=4)
        
    def test_calculate_weight_access_boost(self):
        """测试访问次数对权重的影响"""
        # 测试访问次数增加时的权重变化
        weight = self.mam.calculate_weight(
            current_weight=0.5,
            last_access_time=self.current_time,
            current_time=self.current_time,
            access_count=2
        )
        # 计算预期值：0.5 * (1 + 0.3*2) - 0.05
        expected_weight = 0.5 * 1.6 - 0.05  # 1 + 0.3*2 = 1.6
        self.assertAlmostEqual(weight, expected_weight, places=4)
        
    def test_calculate_weight_bounds(self):
        """测试权重是否始终在[0,1]范围内"""
        # 测试极端情况下的权重范围
        weight = self.mam.calculate_weight(
            current_weight=1.0,
            last_access_time=self.current_time - timedelta(days=30),  # 30天前
            current_time=self.current_time,
            access_count=100  # 大量访问
        )
        self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(weight, 1.0)
        
    def test_select_activated_memories_empty(self):
        """测试空记忆列表的选择"""
        memories = []
        activated = self.mam.select_activated_memories(memories)
        self.assertEqual(len(activated), 0)
        
    def test_select_activated_memories_single(self):
        """测试单个记忆的选择"""
        memories = [{"content": "测试记忆", "weight": 0.8}]
        activated = self.mam.select_activated_memories(memories)
        self.assertEqual(len(activated), 1)
        self.assertEqual(activated[0]["weight"], 0.8)
        
    def test_select_activated_memories_multiple(self):
        """测试多个记忆的选择和黄金分割比例"""
        memories = [
            {"content": "记忆1", "weight": 0.9},
            {"content": "记忆2", "weight": 0.8},
            {"content": "记忆3", "weight": 0.7},
            {"content": "记忆4", "weight": 0.6},
            {"content": "记忆5", "weight": 0.5}
        ]
        activated = self.mam.select_activated_memories(memories)
        # 使用黄金分割比例(0.618)，应该选择前3个记忆
        expected_count = int(len(memories) * self.mam.golden_ratio)
        self.assertEqual(len(activated), expected_count)
        
    def test_select_activated_memories_sorting(self):
        """测试按不同字段排序的记忆选择"""
        memories = [
            {"content": "记忆1", "weight": 0.5, "importance": 3},
            {"content": "记忆2", "weight": 0.8, "importance": 1},
            {"content": "记忆3", "weight": 0.6, "importance": 2}
        ]
        # 按importance排序
        activated = self.mam.select_activated_memories(memories, sort_by="importance")
        self.assertEqual(activated[0]["importance"], 3)
        
    def test_select_activated_memories_missing_field(self):
        """测试排序字段不存在的情况"""
        memories = [
            {"content": "记忆1", "weight": 0.8},
            {"content": "记忆2"},  # 缺少weight字段
            {"content": "记忆3", "weight": 0.6}
        ]
        activated = self.mam.select_activated_memories(memories)
        # 激活列表中不应包含缺失weight字段的元素
        for m in activated:
            self.assertIn("weight", m)

if __name__ == '__main__':
    unittest.main() 