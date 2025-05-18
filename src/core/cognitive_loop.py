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
from ..utils.prompts_config import (
    LLM_CONFIG,
    COGNITIVE_LOOP_CONFIG,
    VECTOR_STORE_CONFIG,
    get_default_llm_model,
    get_system_prompt,
    build_context
)

class CognitiveLoop:
    """认知循环系统
    
    实现认元激活、权重更新、上下文构造等核心功能
    """
    
    def __init__(
        self,
        llm_provider: str = None,
        llm_model: Optional[str] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        b: Optional[float] = None,
        prompt_type: str = "DEFAULT"
    ):
        """初始化认知循环
        
        Args:
            llm_provider: LLM 服务提供商
            llm_model: LLM 模型名称
            beta: 旧记忆残留因子
            gamma: 新调用强化增益系数
            b: 时间敏感系数
            prompt_type: 系统提示词类型
        """
        # 配置 litellm
        litellm.api_key = OPENAI_API_KEY
        
        # 使用配置文件中的默认值
        self.llm_provider = llm_provider or LLM_CONFIG["DEFAULT_PROVIDER"]
        self.llm_model = llm_model or get_default_llm_model(self.llm_provider)
        self.prompt_type = prompt_type
        self.system_prompt = get_system_prompt(prompt_type)
        
        # 初始化向量存储
        self.vector_store = UpstashVectorStore(
            vector_url=UPSTASH_VECTOR_URL,
            vector_token=UPSTASH_VECTOR_TOKEN,
            embedding_model=VECTOR_STORE_CONFIG["DEFAULT_EMBEDDING_MODEL"]
        )
        
        # 使用配置文件中的参数初始化权重更新器
        beta = beta if beta is not None else COGNITIVE_LOOP_CONFIG["beta"]
        gamma = gamma if gamma is not None else COGNITIVE_LOOP_CONFIG["gamma"]
        b = b if b is not None else COGNITIVE_LOOP_CONFIG["b"]
        self.weight_updater = MAMWeightUpdater(beta, gamma, b)
        
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
            top_k=COGNITIVE_LOOP_CONFIG["default_top_k"],
            min_score=COGNITIVE_LOOP_CONFIG["default_min_score"]
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
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ],
            **LLM_CONFIG["DEFAULT_PARAMETERS"]
        )
        
        # 5. 创建新的认元
        await self.vector_store.add_coglet(
            content=input_text,
            weight=1.0,
            timestamp=current_time,
            collection_id="user_input"
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
            # 没有相关记忆，直接使用简化版模板
            return build_context([], input_text)
            
        # 按权重排序
        sorted_coglets = sorted(
            similar_coglets,
            key=lambda x: x[1]["weight"],
            reverse=True
        )
        
        # 转换为 build_context 所需格式
        memories = []
        for coglet_id, metadata, score in sorted_coglets:
            memories.append({
                "content": metadata["content"],
                "weight": metadata["weight"],
                "score": score
            })
            
        # 使用统一的上下文构建函数
        return build_context(memories, input_text) 