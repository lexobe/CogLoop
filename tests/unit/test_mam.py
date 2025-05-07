"""
MAM 模型的单元测试
"""
import unittest
import numpy as np
from src.utils.weight_update import MAMWeightUpdater

class TestMAMWeightUpdater(unittest.TestCase):
    """MAM 权重更新器的测试用例"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 使用文档中推荐的默认参数
        self.updater = MAMWeightUpdater(beta=0.8, gamma=1.0, b=0.1)
        
    def test_initialization(self):
        """测试初始化参数"""
        self.assertEqual(self.updater.beta, 0.8)
        self.assertEqual(self.updater.gamma, 1.0)
        self.assertEqual(self.updater.b, 0.1)
        
    def test_update_weight_zero_delta(self):
        """测试时间间隔为 0 时的权重更新"""
        # 当 Δt = 0 时，权重应该接近 W_t * β
        current_weight = 1.0
        time_delta = 0.0
        new_weight = self.updater.update_weight(current_weight, time_delta)
        expected_weight = current_weight * self.updater.beta
        self.assertAlmostEqual(new_weight, expected_weight, places=5)
        
    def test_update_weight_optimal_interval(self):
        """测试最优时间间隔的权重更新"""
        # 当 Δt = 1/b 时，应该获得最大增益
        current_weight = 1.0
        time_delta = self.updater.get_optimal_interval()  # 1/b
        new_weight = self.updater.update_weight(current_weight, time_delta)
        
        # 验证权重确实增加了
        self.assertGreater(new_weight, current_weight)
        
    def test_update_weight_large_delta(self):
        """测试大时间间隔的权重更新"""
        # 当 Δt 很大时，权重应该接近 0
        current_weight = 1.0
        time_delta = 100.0  # 一个很大的时间间隔
        new_weight = self.updater.update_weight(current_weight, time_delta)
        self.assertLess(new_weight, 0.01)  # 权重应该很小
        
    def test_batch_update(self):
        """测试批量权重更新"""
        weights = [1.0, 2.0, 3.0]
        time_deltas = [0.0, 1.0, 10.0]
        new_weights = self.updater.update_weights_batch(weights, time_deltas)
        
        # 验证返回列表长度
        self.assertEqual(len(new_weights), len(weights))
        
        # 验证每个权重的更新
        for i, (w, dt) in enumerate(zip(weights, time_deltas)):
            expected = self.updater.update_weight(w, dt)
            self.assertAlmostEqual(new_weights[i], expected, places=5)
            
    def test_parameter_effects(self):
        """测试不同参数对权重更新的影响"""
        # 测试不同的 beta 值
        updater_high_beta = MAMWeightUpdater(beta=0.99, gamma=1.0, b=0.1)
        updater_low_beta = MAMWeightUpdater(beta=0.7, gamma=1.0, b=0.1)
        
        current_weight = 1.0
        time_delta = 1.0
        
        weight_high_beta = updater_high_beta.update_weight(current_weight, time_delta)
        weight_low_beta = updater_low_beta.update_weight(current_weight, time_delta)
        
        # beta 越大，保留的历史权重越多
        self.assertGreater(weight_high_beta, weight_low_beta)
        
        # 测试不同的 gamma 值
        updater_high_gamma = MAMWeightUpdater(beta=0.8, gamma=5.0, b=0.1)
        updater_low_gamma = MAMWeightUpdater(beta=0.8, gamma=0.5, b=0.1)
        
        weight_high_gamma = updater_high_gamma.update_weight(current_weight, time_delta)
        weight_low_gamma = updater_low_gamma.update_weight(current_weight, time_delta)
        
        # gamma 越大，新调用的增益越大
        self.assertGreater(weight_high_gamma, weight_low_gamma)
        
    def test_optimal_interval_calculation(self):
        """测试最优时间间隔的计算"""
        # 对于 b = 0.1，最优间隔应该是 10
        self.assertAlmostEqual(self.updater.get_optimal_interval(), 10.0, places=5)
        
        # 测试不同的 b 值
        updater_high_b = MAMWeightUpdater(beta=0.8, gamma=1.0, b=0.2)
        updater_low_b = MAMWeightUpdater(beta=0.8, gamma=1.0, b=0.01)
        
        self.assertAlmostEqual(updater_high_b.get_optimal_interval(), 5.0, places=5)
        self.assertAlmostEqual(updater_low_b.get_optimal_interval(), 100.0, places=5)

if __name__ == '__main__':
    unittest.main() 