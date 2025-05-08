"""
向量数据库服务模块，使用 Upstash Vector 实现
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import logging
import json
from upstash_vector import Index
from upstash_vector.errors import UpstashError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'  # 设置日志编码为 UTF-8
)
logger = logging.getLogger(__name__)

# 黄金分割比例
GOLDEN_RATIO = 0.618

# 环境变量配置
UPSTASH_VECTOR_URL = os.getenv("UPSTASH_VECTOR_URL")
UPSTASH_VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def check_environment() -> Tuple[bool, List[str]]:
    """检查必要的环境变量是否已设置
    
    Returns:
        (是否所有变量都已设置, 缺失的变量列表)
    """
    required_vars = {
        "UPSTASH_VECTOR_URL": UPSTASH_VECTOR_URL,
        "UPSTASH_VECTOR_TOKEN": UPSTASH_VECTOR_TOKEN
    }
    
    missing_vars = [
        var_name for var_name, var_value in required_vars.items()
        if not var_value
    ]
    
    all_set = len(missing_vars) == 0
    return all_set, missing_vars

class UpstashVectorStore:
    """Upstash Vector 存储服务"""

    def __init__(
        self,
        vector_url: Optional[str] = None,
        vector_token: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5"
    ):
        """初始化向量数据库服务
        
        Args:
            vector_url: Upstash Vector API URL，如果为 None 则从环境变量获取
            vector_token: Upstash Vector API Token，如果为 None 则从环境变量获取
            embedding_model: 嵌入模型名称，默认使用 BAAI/bge-small-en-v1.5
        """
        # 初始化 Upstash Vector 配置
        self.vector_url = vector_url or UPSTASH_VECTOR_URL
        self.vector_token = vector_token or UPSTASH_VECTOR_TOKEN
        self.embedding_model = embedding_model

        if not self.vector_url or not self.vector_token:
            logger.error("未配置 Upstash Vector 凭证")
            raise ValueError("Upstash Vector 凭证未配置")

        # 初始化 Upstash Vector 客户端
        try:
            # 使用 Upstash 的内置嵌入功能
            self.index = Index(
                url=self.vector_url,
                token=self.vector_token
            )
            logger.info(f"已连接到 Upstash Vector 服务器: {self.vector_url}")
        except Exception as e:
            logger.error(f"连接 Upstash Vector 服务器失败: {str(e)}")
            raise

    async def check_health(self) -> Tuple[bool, str]:
        """检查 Upstash 服务器是否可用
        
        Returns:
            (是否可用, 状态信息)
        """
        try:
            logger.info("正在检查 Upstash 服务器状态...")
            
            # 1. 检查连接
            if not self.index:
                return False, "Upstash 客户端未初始化"
                
            # 2. 检查 API 凭证
            if not self.vector_url or not self.vector_token:
                return False, "Upstash 凭证未正确配置"
                
            # 3. 尝试执行一个简单的查询操作
            # 使用非零向量进行测试，避免余弦相似度未定义的问题
            try:
                # 直接使用文本查询进行测试
                results = self.index.query(
                    data="健康检查测试",
                    top_k=1,
                    include_metadata=True
                )
                
                # 检查结果是否是列表
                if not isinstance(results, list):
                    return False, f"Upstash 查询返回异常结果: {type(results)}"
                
                return True, "Upstash 服务器运行正常"
            except Exception as e:
                logger.error(f"执行测试查询失败: {str(e)}")
                return False, f"执行测试查询失败: {str(e)}"
                
        except Exception as e:
            logger.error(f"检查 Upstash 服务器状态失败: {str(e)}")
            return False, f"检查服务器状态失败: {str(e)}"

    async def add_coglet(
        self,
        content: str,
        weight: float,
        timestamp: float,
        collection_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加认元到向量数据库
        
        Args:
            content: 认元内容
            weight: 认元权重
            timestamp: 时间戳
            collection_id: 认元集合ID
            metadata: 元数据
            
        Returns:
            认元ID
        """
        try:
            # 准备元数据
            coglet_metadata = {
                "content": content,
                "weight": weight,
                "timestamp": timestamp,
                "collection_id": collection_id,
                **(metadata or {})
            }
            
            # 准备认元ID
            coglet_id = f"{collection_id}:{timestamp}"
            
            # 存储到 Upstash Vector，直接使用文本内容
            self.index.upsert([
                {
                    "id": coglet_id,
                    "data": content,  # 直接使用文本内容
                    "metadata": coglet_metadata
                }
            ])
            
            logger.info(f"已添加认元 {coglet_id} 到 Upstash Vector")
            return coglet_id
        except Exception as e:
            logger.error(f"添加认元到 Upstash Vector 失败: {str(e)}")
            raise

    async def search_similar(
        self,
        query: str,
        collection_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """搜索相似认元
        
        返回 top_k 个结果中相似度分数排名前 61.8%（黄金分割比例）的认元。
        例如，如果 top_k=10，则返回相似度分数最高的 6 个结果。
        
        Args:
            query: 查询文本
            collection_id: 认元集合ID，如果指定则只搜索该集合
            top_k: 返回结果数量
            min_score: 最小相似度分数
            
        Returns:
            认元列表，每个元素为 (id, metadata, score)
        """
        try:
            # 准备查询参数
            query_params = {
                "data": query,  # 直接使用文本查询
                "top_k": top_k,
                "include_metadata": True
            }
            
            # 添加元数据过滤条件
            if collection_id:
                query_params["filter"] = f"collection_id = '{collection_id}'"
            
            # 查询
            results = self.index.query(**query_params)
            
            if not results:
                return []
                
            # 确保结果是列表类型
            if not isinstance(results, list):
                logger.error(f"查询结果类型异常: {type(results)}")
                return []

            # 计算要返回的结果数量（使用黄金分割比例）
            num_results = max(1, int(len(results) * GOLDEN_RATIO))
            
            # 过滤并格式化结果
            filtered_results = []
            for i, result in enumerate(results):
                if i >= num_results:
                    break
                if result.score >= min_score:
                    filtered_results.append((
                        result.id,
                        result.metadata,
                        result.score
                    ))
            
            return filtered_results
        except Exception as e:
            logger.error(f"搜索相似认元失败: {str(e)}")
            raise

    async def query(
        self,
        text: str,
        top_k: int = 1,
        include_metadata: bool = True,
        filter: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """执行文本查询
        
        Args:
            text: 查询文本
            top_k: 返回结果数量
            include_metadata: 是否包含元数据
            filter: 元数据过滤条件
            namespace: 命名空间
            
        Returns:
            查询结果列表
        """
        try:
            # 准备查询参数
            query_params = {
                "data": text,  # 直接使用文本查询
                "top_k": top_k,
                "include_metadata": include_metadata
            }
            
            # 添加可选参数
            if filter:
                query_params["filter"] = filter
            if namespace:
                query_params["namespace"] = namespace
                
            # 执行查询
            results = self.index.query(**query_params)
            
            # 转换结果为列表
            return [
                {
                    "id": result.id,
                    "metadata": result.metadata,
                    "score": result.score
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"执行查询失败: {str(e)}")
            raise

async def main():
    store = UpstashVectorStore(
        vector_url=UPSTASH_VECTOR_URL,
        vector_token=UPSTASH_VECTOR_TOKEN
    )
    
    # 检查服务器状态
    is_healthy, message = await store.check_health()
    print(f"服务器状态: {'✅ 正常' if is_healthy else '❌ 异常'}")
    print(f"状态信息: {message}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 