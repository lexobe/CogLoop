"""
LLM 服务模块的单元测试
"""
import os
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from src.utils.llm_service import (
    LLMResponse,
    UnifiedLLMService,
    LLMServiceFactory
)
from src.utils.config import OPENAI_API_KEY

class TestLLMService(unittest.TestCase):
    """LLM 服务测试用例"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.api_key = OPENAI_API_KEY or "test_api_key"
        
    @pytest.mark.asyncio
    async def test_llm_service_factory(self):
        """测试 LLM 服务工厂"""
        # 测试创建不同提供商的服务
        providers = ["openai", "anthropic", "deepseek", "gemini"]
        for provider in providers:
            service = LLMServiceFactory.create(provider, self.api_key)
            self.assertIsInstance(service, UnifiedLLMService)
            self.assertEqual(service.provider, provider)
            
    @pytest.mark.asyncio
    async def test_generate(self):
        """测试文本生成"""
        # 模拟 litellm 响应
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="测试响应"))
        ]
        mock_response.usage = {"total_tokens": 10}
        mock_response.model_dump = MagicMock(return_value={"test": "data"})
        
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            service = UnifiedLLMService("openai", self.api_key)
            response = await service.generate("测试提示")
            
            self.assertIsInstance(response, LLMResponse)
            self.assertEqual(response.text, "测试响应")
            self.assertEqual(response.usage["total_tokens"], 10)
            
    @pytest.mark.asyncio
    async def test_embed(self):
        """测试文本嵌入"""
        # 模拟 litellm 响应
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            service = UnifiedLLMService("openai", self.api_key)
            embedding = await service.embed("测试文本")
            
            self.assertIsInstance(embedding, list)
            self.assertEqual(len(embedding), 3)
            self.assertEqual(embedding, [0.1, 0.2, 0.3])
            
    def test_default_models(self):
        """测试默认模型配置"""
        service = UnifiedLLMService("openai", self.api_key)
        self.assertEqual(service.model, "gpt-4-turbo-preview")
        
        service = UnifiedLLMService("anthropic", self.api_key)
        self.assertEqual(service.model, "claude-3-opus-20240229")
        
        service = UnifiedLLMService("deepseek", self.api_key)
        self.assertEqual(service.model, "deepseek-chat")
        
        service = UnifiedLLMService("gemini", self.api_key)
        self.assertEqual(service.model, "gemini-pro")
        
        # 测试自定义模型
        custom_model = "custom-model"
        service = UnifiedLLMService("openai", self.api_key, custom_model)
        self.assertEqual(service.model, custom_model)

if __name__ == '__main__':
    unittest.main() 