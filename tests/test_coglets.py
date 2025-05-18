"""
测试 Coglets 类的功能
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
from src.coglets import Coglets
from src.vector_store import VectorStore
from src.log_config import setup_logger
import os

# 设置日志级别
os.environ["LOG_LEVEL_COGLETS"] = "INFO"
logger = setup_logger("Coglets")

@pytest.fixture
def mock_vector_store():
    mock = Mock()
    mock.add_coglet.return_value = "test_id_123"
    mock.get_coglet.return_value = {
        "content": "这是一个测试认元",
        "metadata": {
            "weight": 0.5,
            "last_access": "2024-05-17T22:39:01",
            "access_count": 0
        }
    }
    mock.clean_coglet_set.return_value = True
    return mock

@pytest.fixture
def coglets(mock_vector_store):
    return Coglets(mock_vector_store)

def test_create_set(coglets):
    """测试创建认元集合"""
    # 测试成功创建
    assert coglets.create_set("test_set")
    
    # 测试创建失败
    mock_vector_store = Mock()
    mock_vector_store.clean_coglet_set.side_effect = Exception("清理失败")
    coglets = Coglets(mock_vector_store)
    assert not coglets.create_set("test_set")

def test_add(coglets):
    """测试添加认元"""
    set_id = "test_set"
    content = "这是一个测试认元"
    assert coglets.create_set(set_id)
    coglet_id = coglets.add(set_id, content)
    assert coglet_id == "test_id_123"
    coglet = coglets.get(coglet_id)
    assert coglet["content"] == content
    assert "weight" in coglet["metadata"]
    assert "last_access" in coglet["metadata"]
    assert "access_count" in coglet["metadata"]

def test_add_batch(coglets):
    """测试批量添加认元"""
    # 准备测试数据
    set_id = "test_set"
    items = [
        {"content": "内容1", "metadata": {"tag": "A"}},
        {"content": "内容2", "metadata": {"tag": "B"}}
    ]
    
    # 模拟返回值
    expected_ids = ["id1", "id2"]
    mock_vector_store = Mock()
    mock_vector_store.add_coglets.return_value = expected_ids
    coglets = Coglets(mock_vector_store)
    
    # 测试批量添加
    result = coglets.add_batch(set_id, items)
    
    # 验证结果
    assert result == expected_ids
    mock_vector_store.add_coglets.assert_called_once()
    
    # 验证调用参数
    args, kwargs = mock_vector_store.add_coglets.call_args
    assert args[0] == set_id  # 第一个位置参数是 set_id
    assert len(args[1]) == 2  # 第二个位置参数是 items 列表

def test_update_weight(coglets):
    """测试更新认元权重"""
    # 准备测试数据
    coglet_id = "test_id"
    current_time = datetime.now()
    last_access = current_time - timedelta(hours=1)
    
    # 模拟 get_coglet 返回值
    mock_vector_store = Mock()
    mock_vector_store.get_coglet.return_value = {
        "content": "测试内容",
        "metadata": {
            "weight": 0.5,
            "last_access": last_access.isoformat(),
            "access_count": 1
        }
    }
    coglets = Coglets(mock_vector_store)
    
    # 模拟更新成功
    mock_vector_store.update_coglet.return_value = True
    
    # 测试更新权重
    result = coglets.update_weight(coglet_id, current_time)
    
    # 验证结果
    assert result
    mock_vector_store.update_coglet.assert_called_once()
    
    # 验证调用参数
    args, kwargs = mock_vector_store.update_coglet.call_args
    assert args[0] == coglet_id  # 第一个位置参数是 coglet_id
    metadata_arg = args[1]  # 第二个位置参数是 metadata
    assert "weight" in metadata_arg
    assert "last_access" in metadata_arg
    assert "access_count" in metadata_arg

def test_recall(coglets):
    """测试回忆认元"""
    # 准备测试数据
    set_id = "test_set"
    query = "测试查询"
    top_k = 5
    
    # 模拟搜索结果
    mock_results = [
        {
            "coglet_id": "id1",
            "score": 0.9,
            "content": "内容1",
            "metadata": {"weight": 0.8}
        },
        {
            "coglet_id": "id2",
            "score": 0.7,
            "content": "内容2",
            "metadata": {"weight": 0.6}
        }
    ]
    mock_vector_store = Mock()
    mock_vector_store.search_similar.return_value = mock_results
    coglets = Coglets(mock_vector_store)
    
    # 测试回忆
    result = coglets.recall(set_id, query, top_k)
    
    # 验证结果
    assert "all_results" in result
    assert "activated" in result
    assert result["all_results"] == mock_results
    
    # 验证调用参数
    args, kwargs = mock_vector_store.search_similar.call_args
    assert args[0] == set_id  # 第一个位置参数是 set_id
    assert args[1] == query   # 第二个位置参数是 query
    assert args[2] == top_k   # 第三个位置参数是 top_k

def test_delete(coglets):
    """测试删除认元"""
    # 准备测试数据
    coglet_id = "test_id"
    
    # 模拟删除成功
    mock_vector_store = Mock()
    mock_vector_store.delete_coglet.return_value = True
    coglets = Coglets(mock_vector_store)
    
    # 测试删除
    result = coglets.delete(coglet_id)
    
    # 验证结果
    assert result
    mock_vector_store.delete_coglet.assert_called_once_with(coglet_id)

def test_delete_batch(coglets):
    """测试批量删除认元"""
    # 准备测试数据
    coglet_ids = ["id1", "id2", "id3"]
    
    # 模拟删除成功
    mock_vector_store = Mock()
    mock_vector_store.delete_coglets.return_value = True
    coglets = Coglets(mock_vector_store)
    
    # 测试批量删除
    result = coglets.delete_batch(coglet_ids)
    
    # 验证结果
    assert result
    mock_vector_store.delete_coglets.assert_called_once_with(coglet_ids)

def test_clean_set(coglets):
    """测试清理认元集合"""
    # 准备测试数据
    set_id = "test_set"
    
    # 模拟清理成功
    mock_vector_store = Mock()
    mock_vector_store.clean_coglet_set.return_value = True
    coglets = Coglets(mock_vector_store)
    
    # 测试清理
    result = coglets.clean_set(set_id)
    
    # 验证结果
    assert result
    mock_vector_store.clean_coglet_set.assert_called_once_with(set_id) 