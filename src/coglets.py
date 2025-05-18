"""
认元（Coglet）管理模块

整合向量存储和记忆锚定机制，实现完整的认元管理功能。
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from src.vector_store import VectorStore
from src.mam import MAM
from src.log_config import setup_logger

logger = setup_logger("Coglets")

class Coglets:
    """认元管理类，整合向量存储和记忆锚定机制"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        beta: float = 0.85,      # 时间衰减系数
        gamma: float = 0.3,      # 访问增强系数
        b: float = 0.05,         # 基础衰减率
        initial_weight: float = 0.5,  # 初始权重
        golden_ratio: float = 0.618,  # 黄金分割比例
        top_k: int = 10,  # 新增top_k参数
    ):
        """
        初始化认元管理器
        
        Args:
            vector_store: 向量存储实例
            beta: 时间衰减系数
            gamma: 访问增强系数
            b: 基础衰减率
            initial_weight: 初始权重
            golden_ratio: 黄金分割比例
            top_k: 返回结果数量
        """
        self.vector_store = vector_store
        self.mam = MAM(
            beta=beta,
            gamma=gamma,
            b=b,
            initial_weight=initial_weight,
            golden_ratio=golden_ratio
        )
        self.top_k = top_k
        
    def create_set(self, set_id: str, description: str = "") -> bool:
        """
        创建新的认元集合
        
        Args:
            set_id: 集合ID
            description: 集合描述
            
        Returns:
            bool: 是否创建成功
        """
        try:
            logger.info(f"Create set: {set_id}, desc: {description}")
            self.vector_store.clean_coglet_set(set_id)
            return True
        except Exception as e:
            logger.error(f"Create set failed: {str(e)}")
            return False
            
    def add(
        self,
        set_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加新的认元
        
        Args:
            set_id: 集合ID
            content: 认元内容
            metadata: 认元元数据
            
        Returns:
            str: 认元ID
        """
        # 准备元数据
        full_metadata = {
            "weight": self.mam.initial_weight,
            "last_access": datetime.now().isoformat(),
            "access_count": 0
        }
        
        # 合并自定义元数据
        if metadata:
            full_metadata.update(metadata)
            
        # 添加认元
        logger.info(f"Add coglet: {set_id}, preview: {content[:30]}")
        return self.vector_store.add_coglet(set_id, content, full_metadata)
        
    def add_batch(
        self,
        set_id: str,
        items: List[Dict[str, Any]]
    ) -> List[str]:
        """
        批量添加认元
        
        Args:
            set_id: 集合ID
            items: 认元列表，每个项目包含 content 和可选的 metadata
            
        Returns:
            List[str]: 认元ID列表
        """
        # 准备批量添加的数据
        batch_items = []
        for item in items:
            metadata = {
                "weight": self.mam.initial_weight,
                "last_access": datetime.now().isoformat(),
                "access_count": 0
            }
            if "metadata" in item:
                metadata.update(item["metadata"])
                
            batch_items.append({
                "content": item["content"],
                "metadata": metadata
            })
            
        return self.vector_store.add_coglets(set_id, batch_items)
        
    def get(self, coglet_id: str) -> Dict[str, Any]:
        """
        获取认元信息
        
        Args:
            coglet_id: 认元ID
            
        Returns:
            Dict[str, Any]: 认元信息
        """
        return self.vector_store.get_coglet(coglet_id)
        
    def update_weight(
        self,
        coglet_id: str,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        更新认元权重
        
        Args:
            coglet_id: 认元ID
            current_time: 当前时间，默认为系统当前时间
            
        Returns:
            bool: 是否更新成功
        """
        try:
            coglet = self.get(coglet_id)
            metadata = coglet["metadata"]
            current_time = current_time or datetime.now()
            last_access = datetime.fromisoformat(metadata["last_access"])
            new_weight = self.mam.calculate_weight(
                current_weight=float(metadata["weight"]),
                last_access_time=last_access,
                current_time=current_time,
                access_count=int(metadata["access_count"])
            )
            updated_metadata = {
                "weight": new_weight,
                "last_access": current_time.isoformat(),
                "access_count": int(metadata["access_count"]) + 1
            }
            return self.vector_store.update_coglet(coglet_id, updated_metadata)
        except Exception as e:
            logger.error(f"Update weight failed: {str(e)}")
            return False
            
    def update_weights(
        self,
        coglet_ids: List[str],
        current_time: Optional[datetime] = None
    ) -> List[str]:
        """
        批量更新认元权重
        
        Args:
            coglet_ids: 认元ID列表
            current_time: 当前时间，默认为系统当前时间
            
        Returns:
            List[str]: 成功更新的认元ID列表
        """
        current_time = current_time or datetime.now()
        successful_ids = []
        
        for coglet_id in coglet_ids:
            if self.update_weight(coglet_id, current_time):
                successful_ids.append(coglet_id)
                
        return successful_ids
        
    def recall(
        self,
        set_id: str,
        query: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        回忆认元
        
        Args:
            set_id: 集合ID
            query: 查询文本
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 包含所有结果和激活结果的字典
        """
        logger.info(f"Recall: set={set_id}, query={query}, top_k={self.top_k}")
        results = self.vector_store.search_similar(set_id, query, self.top_k)
        activated = self.mam.select_activated_memories(results)
        logger.info(f"Recall: total={len(results)}, activated={len(activated)}")
        
        return {
            "all_results": results,
            "activated": activated
        }
        
    def delete(self, coglet_id: str) -> bool:
        """
        删除认元
        
        Args:
            coglet_id: 认元ID
            
        Returns:
            bool: 是否删除成功
        """
        return self.vector_store.delete_coglet(coglet_id)
        
    def delete_batch(self, coglet_ids: List[str]) -> bool:
        """
        批量删除认元
        
        Args:
            coglet_ids: 认元ID列表
            
        Returns:
            bool: 是否删除成功
        """
        return self.vector_store.delete_coglets(coglet_ids)
        
    def clean_set(self, set_id: str) -> bool:
        """
        清理认元集合
        
        Args:
            set_id: 集合ID
            
        Returns:
            bool: 是否清理成功
        """
        return self.vector_store.clean_coglet_set(set_id) 