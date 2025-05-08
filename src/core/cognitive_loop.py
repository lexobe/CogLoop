"""
认知循环主逻辑实现
"""
from typing import List, Dict, Any, Optional
import numpy as np
import litellm
from .coglet import Coglet
from ..utils.weight_update import MAMWeightUpdater
from ..utils.vector_store import UpstashVectorStore
from ..utils.config import (
    OPENAI_API_KEY,
    UPSTASH_VECTOR_URL,
    UPSTASH_VECTOR_TOKEN
)

class CognitiveLoop:
    """认知循环系统
    
    实现认元激活、权重更新、上下文构造等核心功能
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        beta: float = 0.8,
        gamma: float = 1.0,
        b: float = 0.1
    ):
        """初始化认知循环
        
        Args:
            llm_provider: LLM 服务提供商
            llm_model: LLM 模型名称
            beta: 旧记忆残留因子
            gamma: 新调用强化增益系数
            b: 时间敏感系数
        """
        # 配置 litellm
        litellm.api_key = OPENAI_API_KEY
        self.llm_provider = llm_provider
        self.llm_model = llm_model or self._get_default_model(llm_provider)
        
        self.vector_store = UpstashVectorStore(
            vector_url=UPSTASH_VECTOR_URL,
            vector_token=UPSTASH_VECTOR_TOKEN
        )
        self.weight_updater = MAMWeightUpdater(beta, gamma, b)
        
    def _get_default_model(self, provider: str) -> str:
        """获取默认模型名称"""
        defaults = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-opus-20240229",
            "deepseek": "deepseek-chat",
            "gemini": "gemini-pro"
        }
        return defaults.get(provider, "gpt-4-turbo-preview")
        
    async def process_input(self, input_text: str) -> str:
        """处理输入文本
        
        Args:
            input_text: 输入文本
            
        Returns:
            处理后的响应文本
        """
        # 1. 搜索相似认元
        similar_coglets = await self.vector_store.search_similar(
            query=input_text,
            top_k=5,
            min_score=0.7
        )
        
        # 2. 更新认元权重
        current_time = np.datetime64('now').astype(np.float64)
        for coglet_id, metadata, _ in similar_coglets:
            time_delta = current_time - metadata["timestamp"]
            new_weight = self.weight_updater.update_weight(
                metadata["weight"],
                time_delta
            )
            await self.vector_store.update_coglet(
                coglet_id=coglet_id,
                weight=new_weight,
                timestamp=current_time
            )
        
        # 3. 构造上下文
        context = self.construct_context(similar_coglets, input_text)
        
        # 4. 生成响应
        response = await litellm.acompletion(
            model=f"{self.llm_provider}/{self.llm_model}",
            messages=[{"role": "user", "content": context}]
        )
        
        # 5. 创建新的认元
        await self.vector_store.add_coglet(
            content=input_text,
            weight=1.0,
            timestamp=current_time
        )
        
        return response.choices[0].message.content
        
    def construct_context(
        self,
        similar_coglets: List[tuple],
        input_text: str
    ) -> str:
        """构造上下文
        
        Args:
            similar_coglets: 相似认元列表，每个元素为 (id, metadata, score)
            input_text: 输入文本
            
        Returns:
            构造的上下文
        """
        if not similar_coglets:
            return input_text
            
        # 按权重排序
        sorted_coglets = sorted(
            similar_coglets,
            key=lambda x: x[1]["weight"],
            reverse=True
        )
        
        # 构造上下文
        context_parts = []
        for _, metadata, score in sorted_coglets:
            context_parts.append(
                f"相关记忆 (权重: {metadata['weight']:.2f}, 相似度: {score:.2f}): "
                f"{metadata['content']}"
            )
            
        context = "\n".join(context_parts)
        context += f"\n\n当前输入: {input_text}"
        
        return context 