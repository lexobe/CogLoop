"""
向量存储服务测试模块
"""
import os
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from upstash_vector.errors import UpstashError
from src.utils.vector_store import UpstashVectorStore, GOLDEN_RATIO

# 测试配置
TEST_VECTOR_URL = "https://test.upstash.io"
TEST_VECTOR_TOKEN = "test_token"

@pytest_asyncio.fixture
async def vector_store():
    """创建向量存储实例"""
    # 保存当前环境变量状态
    original_url = os.environ.get("UPSTASH_VECTOR_URL")
    original_token = os.environ.get("UPSTASH_VECTOR_TOKEN")
    
    # 设置测试环境变量
    os.environ["UPSTASH_VECTOR_URL"] = TEST_VECTOR_URL
    os.environ["UPSTASH_VECTOR_TOKEN"] = TEST_VECTOR_TOKEN
    
    # 创建一个 UpstashVectorStore 的 mock 实例，避免实际连接
    with patch('upstash_vector.Index'):
        store = UpstashVectorStore()
        
        # 替换实际的index为模拟对象
        mock_index = MagicMock()
        # 确保query方法返回的是一个可等待对象
        query_mock = AsyncMock()
        mock_index.query = query_mock
        
        # 确保upsert方法返回的也是一个可等待对象
        upsert_mock = AsyncMock()
        upsert_mock.return_value = "test_id"
        mock_index.upsert = upsert_mock
        
        store.index = mock_index
        
        yield store
    
    # 恢复环境变量
    if original_url:
        os.environ["UPSTASH_VECTOR_URL"] = original_url
    else:
        del os.environ["UPSTASH_VECTOR_URL"]
        
    if original_token:
        os.environ["UPSTASH_VECTOR_TOKEN"] = original_token
    else:
        del os.environ["UPSTASH_VECTOR_TOKEN"]

@pytest.mark.asyncio
class TestUpstashVectorStore:
    """测试 UpstashVectorStore 类"""
    
    async def test_init_with_env_vars(self):
        """测试使用环境变量初始化"""
        # 为src.utils.vector_store模块中的常量打补丁
        with patch('src.utils.vector_store.UPSTASH_VECTOR_URL', TEST_VECTOR_URL), \
             patch('src.utils.vector_store.UPSTASH_VECTOR_TOKEN', TEST_VECTOR_TOKEN), \
             patch('upstash_vector.Index') as mock_index_class:
            
            # 确保创建实例时不会抛出异常
            mock_index_instance = MagicMock()
            mock_index_class.return_value = mock_index_instance
            
            # 创建实例
            store = UpstashVectorStore()
            
            # 验证配置值正确传递
            assert store.vector_url == TEST_VECTOR_URL
            assert store.vector_token == TEST_VECTOR_TOKEN
            assert store.embedding_model == "BAAI/bge-small-en-v1.5"
            assert store.index is not None
        
    async def test_init_with_params(self):
        """测试使用参数初始化"""
        # 创建模拟 Index 类
        with patch('upstash_vector.Index') as mock_index_class:
            # 确保创建实例时不会抛出异常
            mock_index_instance = MagicMock()
            mock_index_class.return_value = mock_index_instance
            
            # 创建实例
            store = UpstashVectorStore(
                vector_url="custom_url",
                vector_token="custom_token",
                embedding_model="custom_model"
            )
            
            # 验证配置
            assert store.vector_url == "custom_url"
            assert store.vector_token == "custom_token"
            assert store.embedding_model == "custom_model"
            assert store.index is not None
        
    async def test_init_missing_required_config(self):
        """测试缺少必要配置"""
        # 保存当前环境变量状态
        original_url = os.environ.get("UPSTASH_VECTOR_URL")
        original_token = os.environ.get("UPSTASH_VECTOR_TOKEN")
        
        try:
            # 清除环境变量
            if "UPSTASH_VECTOR_URL" in os.environ:
                del os.environ["UPSTASH_VECTOR_URL"]
            if "UPSTASH_VECTOR_TOKEN" in os.environ:
                del os.environ["UPSTASH_VECTOR_TOKEN"]
            
            # 为src.utils.vector_store模块中的UPSTASH_VECTOR_URL和UPSTASH_VECTOR_TOKEN打补丁
            with patch('src.utils.vector_store.UPSTASH_VECTOR_URL', None), \
                 patch('src.utils.vector_store.UPSTASH_VECTOR_TOKEN', None):
                
                # 验证异常
                with pytest.raises(ValueError) as exc_info:
                    UpstashVectorStore()
                assert "Upstash Vector 凭证未配置" in str(exc_info.value)
        finally:
            # 恢复环境变量
            if original_url:
                os.environ["UPSTASH_VECTOR_URL"] = original_url
            if original_token:
                os.environ["UPSTASH_VECTOR_TOKEN"] = original_token
    
    async def test_check_health_success(self, vector_store):
        """测试健康检查成功"""
        # 模拟成功的查询响应
        mock_result = MagicMock()
        mock_result.id = "test_id"
        vector_store.index.query.return_value = [mock_result]
        
        # 打补丁替换 index.query 方法
        with patch.object(vector_store.index, 'query', new=AsyncMock(return_value=[mock_result])):
            # 执行测试
            is_healthy, message = await vector_store.check_health()
            
            # 验证结果
            assert is_healthy is True
            assert message == "Upstash 服务器运行正常"
        
    async def test_check_health_upstash_error(self, vector_store):
        """测试健康检查 - Upstash 错误"""
        # 打补丁替换 index.query 方法，模拟 Upstash 错误
        with patch.object(vector_store.index, 'query', 
                         new=AsyncMock(side_effect=UpstashError("连接超时"))):
            # 执行测试
            is_healthy, message = await vector_store.check_health()
            
            # 验证结果
            assert is_healthy is False
            assert "执行测试查询失败" in message
            assert "连接超时" in message
        
    async def test_check_health_unknown_error(self, vector_store):
        """测试健康检查 - 未知错误"""
        # 打补丁替换 index.query 方法，模拟未知错误
        with patch.object(vector_store.index, 'query', 
                         new=AsyncMock(side_effect=Exception("未知错误"))):
            # 执行测试
            is_healthy, message = await vector_store.check_health()
            
            # 验证结果
            assert is_healthy is False
            assert "执行测试查询失败" in message
            assert "未知错误" in message
    
    async def test_add_coglet(self, vector_store):
        """测试添加认元"""
        # 设置测试数据
        test_content = "test content"
        test_weight = 1.0
        test_timestamp = 123.45
        test_collection_id = "test_collection"
        expected_id = f"{test_collection_id}:{test_timestamp}"
        
        # 打补丁替换 index.upsert 方法
        with patch.object(vector_store.index, 'upsert', 
                         new=AsyncMock(return_value=expected_id)):
            # 执行测试
            coglet_id = await vector_store.add_coglet(
                content=test_content,
                weight=test_weight,
                timestamp=test_timestamp,
                collection_id=test_collection_id
            )
            
            # 验证结果
            assert coglet_id == expected_id
            
            # 获取调用参数
            call_args = vector_store.index.upsert.call_args
            args, _ = call_args
            upsert_data = args[0][0]
            
            # 验证数据
            assert upsert_data["id"] == expected_id
            assert upsert_data["data"] == test_content
            assert upsert_data["metadata"]["content"] == test_content
            assert upsert_data["metadata"]["weight"] == test_weight
            assert upsert_data["metadata"]["timestamp"] == test_timestamp
            assert upsert_data["metadata"]["collection_id"] == test_collection_id
            
    async def test_search_similar(self, vector_store):
        """测试搜索相似认元"""
        # 模拟搜索结果
        mock_results = []
        for i in range(10):  # 创建10个测试结果
            mock_result = MagicMock()
            mock_result.id = f"test_id_{i}"
            mock_result.metadata = {
                "content": f"test content {i}",
                "collection_id": "test_collection"
            }
            # 分数从0.95递减到0.75，确保所有分数都大于min_score（0.7）
            mock_result.score = 0.95 - (i * 0.02)
            mock_results.append(mock_result)
        
        # 打补丁替换 index.query 方法
        with patch.object(vector_store.index, 'query', 
                         new=AsyncMock(return_value=mock_results)):
            # 执行测试
            results = await vector_store.search_similar(
                query="test query",
                collection_id="test_collection",
                top_k=10
            )
            
            # 验证结果数量
            expected_num_results = max(1, int(10 * GOLDEN_RATIO))  # 应该是6个结果
            assert len(results) == expected_num_results
            
            # 验证结果按分数排序
            for i in range(len(results) - 1):
                assert results[i][2] >= results[i + 1][2]
                
            # 验证所有结果的分数都大于等于最小分数
            for result in results:
                assert result[2] >= 0.7  # min_score 默认值
            
            # 验证调用
            call_args = vector_store.index.query.call_args
            args, kwargs = call_args
            assert kwargs["data"] == "test query"
            assert kwargs["top_k"] == 10
            assert kwargs["filter"] == "collection_id = 'test_collection'"
            assert kwargs["include_metadata"] is True
            
    async def test_query(self, vector_store):
        """测试直接查询方法"""
        # 模拟搜索结果
        mock_results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.id = f"test_id_{i}"
            mock_result.metadata = {
                "content": f"test content {i}",
                "collection_id": "test_collection"
            }
            mock_result.score = 0.95 - (i * 0.05)
            mock_results.append(mock_result)
        
        # 打补丁替换 index.query 方法
        with patch.object(vector_store.index, 'query', 
                        new=AsyncMock(return_value=mock_results)):
            # 执行测试
            results = await vector_store.query(
                text="test query",
                top_k=3,
                filter="collection_id = 'test_collection'"
            )
            
            # 验证结果
            assert len(results) == 3
            assert results[0]["id"] == "test_id_0"
            assert results[0]["metadata"]["content"] == "test content 0"
            assert results[0]["score"] == 0.95
            
            # 验证调用
            call_args = vector_store.index.query.call_args
            args, kwargs = call_args
            assert kwargs["data"] == "test query"
            assert kwargs["top_k"] == 3
            assert kwargs["filter"] == "collection_id = 'test_collection'"
            assert kwargs["include_metadata"] is True 