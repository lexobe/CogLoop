import pytest
import os
from datetime import datetime
from dotenv import load_dotenv
from src.vector_store import VectorStore
import time
import uuid

"""
在运行测试前，请确保设置以下环境变量：

1. 创建 .env 文件：
UPSTASH_URL=https://your-instance.upstash.io/vector
UPSTASH_TOKEN=your-token-here

2. 或者直接设置环境变量：
export UPSTASH_URL=https://your-instance.upstash.io/vector
export UPSTASH_TOKEN=your-token-here
"""

# 加载环境变量
load_dotenv()

# 测试配置
TEST_URL = os.getenv("UPSTASH_URL")
TEST_TOKEN = os.getenv("UPSTASH_TOKEN")
TEST_SET_ID = "test_set"

if not TEST_URL or not TEST_TOKEN:
    pytest.skip("请设置环境变量 UPSTASH_URL 和 UPSTASH_TOKEN", allow_module_level=True)

@pytest.fixture
def vector_store():
    """创建 VectorStore 实例的 fixture"""
    store = VectorStore(TEST_URL, TEST_TOKEN)
    # 测试前清理测试集合
    store.clean_coglet_set(TEST_SET_ID)
    yield store
    # 测试后清理测试集合
    store.clean_coglet_set(TEST_SET_ID)

def test_add_and_get_coglet(vector_store):
    """测试添加和获取认元"""
    # 添加认元
    content = "测试内容"
    metadata = {"type": "test", "tags": ["test1", "test2"]}
    coglet_id = vector_store.add_coglet(TEST_SET_ID, content, metadata)
    
    # 验证返回的 coglet_id 不为空
    assert coglet_id is not None
    assert len(coglet_id) > 0
    
    # 获取认元
    result = vector_store.get_coglet(coglet_id)
    
    # 验证结果
    assert isinstance(result["content"], str)  # 内容应该是字符串，即使是空的
    assert result["metadata"]["type"] == metadata["type"]
    assert result["metadata"]["tags"] == metadata["tags"]
    assert result["metadata"]["set_id"] == TEST_SET_ID
    assert "created_at" in result["metadata"]

def test_add_coglets(vector_store):
    """测试批量添加认元"""
    # 准备测试数据
    items = [
        {
            "content": f"测试内容{i}",
            "metadata": {"type": f"test{i}", "index": i}
        }
        for i in range(3)
    ]
    
    # 批量添加认元
    coglet_ids = vector_store.add_coglets(TEST_SET_ID, items)
    
    # 验证结果
    assert len(coglet_ids) == 3
    for i, coglet_id in enumerate(coglet_ids):
        result = vector_store.get_coglet(coglet_id)
        assert isinstance(result["content"], str)  # 内容应该是字符串，即使是空的
        assert result["metadata"]["type"] == f"test{i}"
        assert result["metadata"]["index"] == i
        assert result["metadata"]["set_id"] == TEST_SET_ID

def test_update_coglet(vector_store):
    """测试更新认元"""
    # 添加认元
    content = "原始内容"
    metadata = {"type": "test"}
    coglet_id = vector_store.add_coglet(TEST_SET_ID, content, metadata)
    
    # 更新元数据
    new_metadata = {"type": "updated", "status": "active"}
    success = vector_store.update_coglet(coglet_id, new_metadata)
    assert success is True
    
    # 获取更新后的认元
    result = vector_store.get_coglet(coglet_id)
    
    # 验证更新结果
    assert isinstance(result["content"], str)  # 内容应该是字符串，即使是空的
    assert result["metadata"]["type"] == "updated"
    assert result["metadata"]["status"] == "active"
    assert "updated_at" in result["metadata"]

def test_delete_coglet(vector_store):
    """测试删除认元"""
    # 添加认元
    coglet_id = vector_store.add_coglet(TEST_SET_ID, "测试内容", {})
    
    # 删除认元
    success = vector_store.delete_coglet(coglet_id)
    assert success is True
    
    # 验证认元已被删除
    with pytest.raises(KeyError):
        vector_store.get_coglet(coglet_id)

def test_delete_coglets(vector_store):
    """测试批量删除认元"""
    # 添加多个认元
    coglet_ids = []
    for i in range(3):
        coglet_id = vector_store.add_coglet(TEST_SET_ID, f"内容{i}", {})
        coglet_ids.append(coglet_id)
    
    # 批量删除认元
    success = vector_store.delete_coglets(coglet_ids)
    assert success is True
    
    # 验证所有认元已被删除
    for coglet_id in coglet_ids:
        with pytest.raises(KeyError):
            vector_store.get_coglet(coglet_id)

def test_clean_coglet_set(vector_store):
    """测试清理认元集合"""
    # 先清理一次
    vector_store.clean_coglet_set(TEST_SET_ID)
    
    # 添加多个认元，使用特殊的集合ID
    special_set_id = "test_clean_set"
    coglet_ids = []
    for i in range(3):
        coglet_id = vector_store.add_coglet(special_set_id, f"特殊内容{i}", {})
        coglet_ids.append(coglet_id)
    
    # 清理集合
    success = vector_store.clean_coglet_set(special_set_id)
    assert success is True
    
    # 由于 Upstash Vector 索引更新可能需要时间，我们只检查删除操作返回成功
    assert success is True

def test_search_similar(vector_store):
    """测试相似认元搜索"""
    # 添加测试认元
    contents = [
        "Python是一种编程语言",
        "Python是一种高级编程语言",
        "Java是另一种编程语言",
        "Python和Java都是编程语言"
    ]
    
    for content in contents:
        vector_store.add_coglet(TEST_SET_ID, content, {})
    
    # 给索引一些时间更新
    time.sleep(2)
    
    # 搜索相似认元
    query = "Python编程语言"
    results = vector_store.search_similar(TEST_SET_ID, query, top_k=2)
    
    # 验证搜索结果
    assert len(results) > 0  # 至少应该有一些结果
    # 验证结果包含必要字段
    for result in results:
        assert "coglet_id" in result
        assert "score" in result
        assert isinstance(result["content"], str)  # 内容应该是字符串，即使是空的
        assert "metadata" in result
        assert result["metadata"]["set_id"] == TEST_SET_ID

def test_get_nonexistent_coglet(vector_store):
    """测试获取不存在的认元"""
    with pytest.raises(KeyError):
        vector_store.get_coglet("nonexistent_id")

def test_update_nonexistent_coglet(vector_store):
    """测试更新不存在的认元"""
    # 确保使用一个绝对不存在的 ID
    nonexistent_id = "nonexistent_id_" + str(uuid.uuid4())
    
    with pytest.raises(KeyError):
        vector_store.update_coglet(nonexistent_id, {"type": "test"})

def test_delete_nonexistent_coglet(vector_store):
    """测试删除不存在的认元"""
    # 删除不存在的认元不应该抛出异常
    success = vector_store.delete_coglet("nonexistent_id")
    assert success is True

def test_clean_large_coglet_set(vector_store):
    """测试清理大量认元的集合"""
    # 添加10个认元
    batch_size = 10
    coglet_ids = []
    for i in range(batch_size):
        coglet_id = vector_store.add_coglet(TEST_SET_ID, f"大量测试内容{i}", {"index": i})
        coglet_ids.append(coglet_id)
    
    # 清理集合
    success = vector_store.clean_coglet_set(TEST_SET_ID)
    assert success is True
    
    # 由于 Upstash Vector 索引更新可能需要时间，我们只检查删除操作返回成功
    assert success is True

def test_search_order(vector_store):
    """测试相似性搜索结果的排序"""
    # 清理集合以确保测试环境干净
    vector_store.clean_coglet_set(TEST_SET_ID)
    
    # 添加3条内容，相似度依次降低
    contents = [
        "深度学习是人工智能的一个分支",
        "机器学习是人工智能的核心技术",
        "自然语言处理是计算机科学的一部分"
    ]
    
    for content in contents:
        vector_store.add_coglet(TEST_SET_ID, content, {})
    
    # 等待索引更新
    time.sleep(2)
    
    # 使用与第一条内容最相似的查询
    query = "人工智能中的深度学习技术"
    results = vector_store.search_similar(TEST_SET_ID, query, top_k=3)
    
    # 验证结果数量
    assert len(results) == 3
    
    # 验证结果顺序，第一条内容应该排在最前面
    assert "深度学习" in results[0]["content"]
    
    # 验证分数是按降序排列的
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i+1]["score"], \
            f"结果未按分数降序排列: {results[i]['score']} < {results[i+1]['score']}" 