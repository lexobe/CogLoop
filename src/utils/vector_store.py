"""
向量数据库服务模块，使用 Upstash Vector 实现
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import litellm
from upstash_vector import Index
from upstash_redis import Redis
from .llm_service import BaseLLMService

class UpstashVectorStore:
    """Upstash 向量数据库服务"""
    
    def __init__(
        self,
        vector_url: str,
        vector_token: str,
        redis_url: str,
        redis_token: str,
        llm_provider: str = "openai",
        llm_model: str = "text-embedding-3-small",
        index_name: str = "cogloop_index",
        dimension: int = 1536  # OpenAI text-embedding-3-small 的维度
    ):
        """初始化向量数据库服务
        
        Args:
            vector_url: Upstash Vector API URL
            vector_token: Upstash Vector API Token
            redis_url: Upstash Redis API URL
            redis_token: Upstash Redis API Token
            llm_provider: LLM 服务提供商
            llm_model: LLM 模型名称
            index_name: 向量索引名称
            dimension: 向量维度
        """
        self.index = Index(
            url=vector_url,
            token=vector_token,
            index_name=index_name,
            dimension=dimension
        )
        self.redis = Redis(
            url=redis_url,
            token=redis_token
        )
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
    async def add_coglet(
        self,
        content: str,
        weight: float,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加认元到向量数据库
        
        Args:
            content: 认元内容
            weight: 认元权重
            timestamp: 时间戳
            metadata: 元数据
            
        Returns:
            认元ID
        """
        # 生成文本嵌入
        embedding = await self._get_embedding(content)
        
        # 准备元数据
        coglet_metadata = {
            "content": content,
            "weight": weight,
            "timestamp": timestamp,
            **(metadata or {})
        }
        
        # 添加到向量数据库
        vector_id = await self.index.upsert(
            vectors=[(embedding, coglet_metadata)]
        )
        
        # 保存到 Redis 用于快速检索
        await self.redis.hset(
            f"coglet:{vector_id}",
            mapping=coglet_metadata
        )
        
        return vector_id
        
    async def search_similar(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """搜索相似认元
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            min_score: 最小相似度分数
            
        Returns:
            认元列表，每个元素为 (id, metadata, score)
        """
        # 生成查询文本的嵌入
        query_embedding = await self._get_embedding(query)
        
        # 向量搜索
        results = await self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # 过滤并格式化结果
        filtered_results = []
        for result in results:
            if result.score >= min_score:
                filtered_results.append((
                    result.id,
                    result.metadata,
                    result.score
                ))
                
        return filtered_results
        
    async def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            文本嵌入向量
        """
        try:
            response = await litellm.aembedding(
                model=f"{self.llm_provider}/{self.llm_model}",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"获取文本嵌入失败: {str(e)}")
        
    async def update_coglet(
        self,
        coglet_id: str,
        weight: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """更新认元信息
        
        Args:
            coglet_id: 认元ID
            weight: 新权重
            timestamp: 新时间戳
            metadata: 新元数据
        """
        # 获取现有元数据
        existing_metadata = await self.redis.hgetall(f"coglet:{coglet_id}")
        
        # 更新元数据
        if weight is not None:
            existing_metadata["weight"] = weight
        if timestamp is not None:
            existing_metadata["timestamp"] = timestamp
        if metadata:
            existing_metadata.update(metadata)
            
        # 更新 Redis
        await self.redis.hset(
            f"coglet:{coglet_id}",
            mapping=existing_metadata
        )
        
        # 更新向量数据库中的元数据
        await self.index.update_metadata(
            id=coglet_id,
            metadata=existing_metadata
        )
        
    async def delete_coglet(self, coglet_id: str):
        """删除认元
        
        Args:
            coglet_id: 认元ID
        """
        # 从向量数据库中删除
        await self.index.delete([coglet_id])
        
        # 从 Redis 中删除
        await self.redis.delete(f"coglet:{coglet_id}")
        
    async def get_coglet(self, coglet_id: str) -> Optional[Dict[str, Any]]:
        """获取认元信息
        
        Args:
            coglet_id: 认元ID
            
        Returns:
            认元元数据
        """
        return await self.redis.hgetall(f"coglet:{coglet_id}")
        
    async def list_coglets(
        self,
        pattern: str = "coglet:*",
        page_size: int = 100
    ) -> List[Dict[str, Any]]:
        """列出所有认元
        
        Args:
            pattern: 匹配模式
            page_size: 每页大小
            
        Returns:
            认元列表
        """
        cursor = 0
        coglets = []
        
        while True:
            # 扫描 Redis
            cursor, keys = await self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=page_size
            )
            
            # 获取认元数据
            for key in keys:
                coglet_data = await self.redis.hgetall(key)
                if coglet_data:
                    coglets.append(coglet_data)
                    
            # 检查是否完成
            if cursor == 0:
                break
                
        return coglets 