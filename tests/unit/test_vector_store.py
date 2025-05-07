"""
向量存储模块的单元测试
"""
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import numpy as np
from src.utils.vector_store import UpstashVectorStore
from src.utils.llm_service import UnifiedLLMService
from src.utils.config import (
    UPSTASH_VECTOR_URL,
    UPSTASH_VECTOR_TOKEN,
    UPSTASH_REDIS_URL,
    UPSTASH_REDIS_TOKEN,
    OPENAI_API_KEY
)

class TestUpstashVectorStore(unittest.TestCase):
    """Upstash 向量存储测试用例"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.llm_service = UnifiedLLMService(
            provider="openai",
            api_key=OPENAI_API_KEY or "test_api_key"
        )
        self.vector_store = UpstashVectorStore(
            vector_url=UPSTASH_VECTOR_URL or "test_vector_url",
            vector_token=UPSTASH_VECTOR_TOKEN or "test_vector_token",
            redis_url=UPSTASH_REDIS_URL or "test_redis_url",
            redis_token=UPSTASH_REDIS_TOKEN or "test_redis_token",
            llm_service=self.llm_service
        )
        
    @pytest.mark.asyncio
    async def test_add_coglet(self):
        """测试添加认元"""
        # 模拟 LLM 服务
        mock_embedding = [0.1, 0.2, 0.3]
        self.llm_service.embed = AsyncMock(return_value=mock_embedding)
        
        # 模拟 Upstash 响应
        mock_vector_id = "test_id"
        self.vector_store.index.upsert = AsyncMock(return_value=mock_vector_id)
        self.vector_store.redis.hset = AsyncMock()
        
        # 测试添加认元
        coglet_id = await self.vector_store.add_coglet(
            content="测试内容",
            weight=1.0,
            timestamp=1234567890.0
        )
        
        self.assertEqual(coglet_id, mock_vector_id)
        self.vector_store.index.upsert.assert_called_once()
        self.vector_store.redis.hset.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_search_similar(self):
        """测试搜索相似认元"""
        # 模拟 LLM 服务
        mock_embedding = [0.1, 0.2, 0.3]
        self.llm_service.embed = AsyncMock(return_value=mock_embedding)
        
        # 模拟搜索结果
        mock_results = [
            MagicMock(
                id="test_id_1",
                metadata={"content": "测试内容1", "weight": 0.8},
                score=0.9
            ),
            MagicMock(
                id="test_id_2",
                metadata={"content": "测试内容2", "weight": 0.6},
                score=0.7
            )
        ]
        self.vector_store.index.query = AsyncMock(return_value=mock_results)
        
        # 测试搜索
        results = await self.vector_store.search_similar(
            query="测试查询",
            top_k=2,
            min_score=0.7
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "test_id_1")
        self.assertEqual(results[0][1]["content"], "测试内容1")
        self.assertEqual(results[0][2], 0.9)
        
    @pytest.mark.asyncio
    async def test_update_coglet(self):
        """测试更新认元"""
        # 模拟现有元数据
        mock_metadata = {
            "content": "测试内容",
            "weight": 0.8,
            "timestamp": 1234567890.0
        }
        self.vector_store.redis.hgetall = AsyncMock(return_value=mock_metadata)
        self.vector_store.redis.hset = AsyncMock()
        self.vector_store.index.update_metadata = AsyncMock()
        
        # 测试更新
        await self.vector_store.update_coglet(
            coglet_id="test_id",
            weight=0.9,
            timestamp=1234567891.0
        )
        
        self.vector_store.redis.hset.assert_called_once()
        self.vector_store.index.update_metadata.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_delete_coglet(self):
        """测试删除认元"""
        self.vector_store.index.delete = AsyncMock()
        self.vector_store.redis.delete = AsyncMock()
        
        await self.vector_store.delete_coglet("test_id")
        
        self.vector_store.index.delete.assert_called_once_with(["test_id"])
        self.vector_store.redis.delete.assert_called_once_with("coglet:test_id")
        
    @pytest.mark.asyncio
    async def test_get_coglet(self):
        """测试获取认元"""
        mock_metadata = {
            "content": "测试内容",
            "weight": 0.8,
            "timestamp": 1234567890.0
        }
        self.vector_store.redis.hgetall = AsyncMock(return_value=mock_metadata)
        
        result = await self.vector_store.get_coglet("test_id")
        
        self.assertEqual(result, mock_metadata)
        self.vector_store.redis.hgetall.assert_called_once_with("coglet:test_id")
        
    @pytest.mark.asyncio
    async def test_list_coglets(self):
        """测试列出认元"""
        # 模拟 Redis 扫描结果
        mock_keys = ["coglet:1", "coglet:2"]
        self.vector_store.redis.scan = AsyncMock(return_value=(0, mock_keys))
        
        # 模拟认元数据
        mock_metadata = {
            "content": "测试内容",
            "weight": 0.8,
            "timestamp": 1234567890.0
        }
        self.vector_store.redis.hgetall = AsyncMock(return_value=mock_metadata)
        
        results = await self.vector_store.list_coglets()
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], mock_metadata)
        self.assertEqual(results[1], mock_metadata)

if __name__ == '__main__':
    unittest.main() 