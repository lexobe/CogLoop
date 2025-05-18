from typing import Dict, List, Any, Optional
from upstash_vector import Index, Vector
from upstash_vector.types import QueryResult
from datetime import datetime
import json
import time
from src.id_generator import vector_id
from src.log_config import setup_logger

logger = setup_logger("VectorStore")

class VectorStore:
    """向量存储实现，使用 Upstash 作为后端存储"""
    
    def __init__(self, url: str, token: str):
        """
        初始化向量存储
        
        Args:
            url: Upstash 服务地址
            token: Upstash 访问令牌
        """
        self.index = Index(
            url=url,
            token=token
        )
        
    def clean_coglet_set(self, set_id: str) -> bool:
        """
        清理指定认元集合中的所有认元
        
        Args:
            set_id: 认元集合ID
            
        Returns:
            bool: 操作是否成功
        """
        logger.info(f"Clean set: {set_id}")
        batch_size = 1000
        max_attempts = 3
        
        for attempt in range(max_attempts):
            # 查找集合中的认元
            results = self.index.query(
                data="",
                top_k=batch_size,
                filter=f"set_id = '{str(set_id)}'",
                include_metadata=False,
                include_data=False
            )
            
            logger.debug(f"Query {attempt+1}, found: {len(results) if results else 0}")
            
            if not results:
                # 没有找到认元，说明已经清理完成
                logger.info(f"Set {set_id} cleaned")
                return True
                
            # 批量删除找到的认元
            coglet_ids = [str(result.id) for result in results]
            success = self.index.delete(ids=coglet_ids)
            
            logger.debug(f"Batch delete: {coglet_ids}")
            
            # 验证删除成功，等待一段时间以确保索引更新
            time.sleep(2)
            
            # 再次查询以确认删除成功
            check_results = self.index.query(
                data="",
                top_k=batch_size,
                filter=f"set_id = '{str(set_id)}'",
                include_metadata=False,
                include_data=False
            )
            
            logger.debug(f"After delete, left: {len(check_results) if check_results else 0}")
            
            # 如果删除完成或只剩少量认元，继续尝试
            if not check_results or len(check_results) < len(results) / 2:
                continue
            
            # 如果删除效果不明显，等待更长时间后重试
            time.sleep(5)
        
        # 最终确认是否完全删除
        final_check = self.index.query(
            data="",
            top_k=10,
            filter=f"set_id = '{str(set_id)}'",
            include_metadata=False,
            include_data=False
        )
        
        logger.info(f"Final check {set_id}, left: {len(final_check) if final_check else 0}")
        
        return len(final_check) == 0
        
    def add_coglet(self, set_id: str, content: str, metadata: Dict[str, Any]) -> str:
        """
        添加新的认元
        
        Args:
            set_id: 认元集合ID
            content: 认元内容
            metadata: 认元元数据
            
        Returns:
            str: 新认元的ID
        """
        # 生成基于内容的 ID
        coglet_id = vector_id(set_id, content)
        
        logger.info(f"Add coglet: {coglet_id} to set: {set_id}")
        
        # 准备元数据，确保所有值都是基本类型
        full_metadata = {
            "set_id": str(set_id),
            "created_at": datetime.now().isoformat()
        }
        
        # 处理自定义元数据，确保所有值都可以被序列化
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                full_metadata[str(key)] = value
            else:
                # 对于复杂类型，转换为 JSON 字符串
                full_metadata[str(key)] = json.dumps(value)
        
        # 使用 Vector 类构建数据
        vector = Vector(
            id=coglet_id,
            data=str(content),
            metadata=full_metadata
        )
        
        # 添加认元
        self.index.upsert(vectors=[vector])
        logger.debug(f"认元{coglet_id}已写入向量库")
        return coglet_id
        
    def add_coglets(self, set_id: str, items: List[Dict[str, Any]]) -> List[str]:
        """
        批量添加认元
        
        Args:
            set_id: 认元集合ID
            items: 认元列表，每个项目包含 content 和 metadata
            
        Returns:
            List[str]: 新认元的ID列表
        """
        logger.info(f"批量添加认元到集合: {set_id}，数量: {len(items)}")
        vectors = []
        coglet_ids = []
        
        for item in items:
            # 生成基于内容的 ID
            coglet_id = vector_id(set_id, item["content"])
            coglet_ids.append(coglet_id)
            
            # 准备元数据
            full_metadata = {
                "set_id": str(set_id),
                "created_at": datetime.now().isoformat()
            }
            
            # 处理自定义元数据
            if "metadata" in item:
                for key, value in item["metadata"].items():
                    if isinstance(value, (str, int, float, bool)):
                        full_metadata[str(key)] = value
                    else:
                        full_metadata[str(key)] = json.dumps(value)
            
            vectors.append(Vector(
                id=coglet_id,
                data=str(item["content"]),
                metadata=full_metadata
            ))
            
        self.index.upsert(vectors=vectors)
        logger.debug(f"批量认元已写入向量库: {coglet_ids}")
        return coglet_ids
        
    def get_coglet(self, coglet_id: str) -> Dict[str, Any]:
        """
        获取指定认元的信息
        
        Args:
            coglet_id: 认元ID
            
        Returns:
            Dict[str, Any]: 认元信息，包含内容和元数据
        """
        logger.info(f"Get coglet: {coglet_id}")
        result = self.index.fetch(
            ids=[str(coglet_id)],
            include_metadata=True,
            include_data=True
        )
        
        if not result:
            logger.warning(f"Coglet not found: {coglet_id}")
            raise KeyError(f"Coglet {coglet_id} not found")
        
        # 获取第一个结果
        vector_result = result[0]
        if vector_result is None:
            raise KeyError(f"Coglet {coglet_id} not found")
            
        # 获取数据
        content = str(vector_result.data or "")
        metadata = vector_result.metadata or {}
        
        # 处理元数据
        processed_metadata = self._process_metadata(metadata)
        
        return {
            "content": content,
            "metadata": processed_metadata
        }
        
    def update_coglet(self, coglet_id: str, metadata: Dict[str, Any]) -> bool:
        """
        更新认元的元数据
        
        Args:
            coglet_id: 认元ID
            metadata: 新的元数据
        
        Returns:
            bool: 操作是否成功
        """
        logger.info(f"Update coglet: {coglet_id}")
        # 先检查认元是否存在
        result = self.index.fetch(
            ids=[str(coglet_id)],
            include_metadata=True,
            include_data=True
        )
        if not result or result[0] is None:
            logger.error(f"Coglet not found: {coglet_id}")
            raise KeyError(f"Coglet {coglet_id} not found")
        # 自动补充 updated_at 字段
        metadata = dict(metadata)
        metadata["updated_at"] = datetime.now().isoformat()
        try:
            update_result = self.index.update(id=coglet_id, metadata=metadata)
            return bool(update_result)
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            return False
        
    def delete_coglet(self, coglet_id: str) -> bool:
        """
        删除指定的认元
        
        Args:
            coglet_id: 认元ID
        
        Returns:
            bool: 操作是否成功
        """
        logger.info(f"Delete coglet: {coglet_id}")
        try:
            result = self.index.delete(ids=[coglet_id])
            # Upstash 返回 DeleteResult(deleted=1) 或 0，兼容 True/False
            return getattr(result, "deleted", 0) >= 0
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            return False
        
    def delete_coglets(self, coglet_ids: List[str]) -> bool:
        """
        批量删除认元
        
        Args:
            coglet_ids: 认元ID列表
        
        Returns:
            bool: 操作是否成功
        """
        logger.info(f"Batch delete: {coglet_ids}")
        try:
            result = self.index.delete(ids=coglet_ids)
            return getattr(result, "deleted", 0) >= 0
        except Exception as e:
            logger.error(f"Batch delete failed: {str(e)}")
            return False
        
    def search_similar(self, set_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        在指定认元集合中搜索相似认元
        
        Args:
            set_id: 认元集合ID
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            List[Dict[str, Any]]: 相似认元列表
        """
        logger.info(f"Search: set={set_id}, query={query}, top_k={top_k}")
        try:
            # 添加小延迟，确保索引已更新
            time.sleep(0.5)
            # 尝试多次查询以提高成功率
            for attempt in range(2):
                results = self.index.query(
                    data=query,
                    top_k=top_k,
                    filter=f"set_id = '{str(set_id)}'",
                    include_metadata=True,
                    include_data=True
                )
                if results:
                    break
                time.sleep(1)
            logger.debug(f"Search result: {len(results) if results else 0}")
            output = []
            for r in results:
                output.append({
                    "coglet_id": r.id,
                    "id": r.id,
                    "content": r.data,
                    "metadata": self._process_metadata(r.metadata),
                    "score": getattr(r, "score", None)
                })
            return output
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理元数据，尝试将 JSON 字符串转换回原始类型
        
        Args:
            metadata: 原始元数据
            
        Returns:
            Dict[str, Any]: 处理后的元数据
        """
        processed = {}
        for key, value in metadata.items():
            if key in ["set_id", "created_at", "updated_at"]:  # 保留系统元数据
                processed[key] = value
                continue
                
            if isinstance(value, str):
                try:
                    processed[key] = json.loads(value)
                except json.JSONDecodeError:
                    processed[key] = value
            else:
                processed[key] = value
        return processed 