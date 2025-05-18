"""
ID 生成工具模块

提供各种 ID 生成函数，用于不同场景下的唯一标识符生成
"""

import uuid
import hashlib


def generate_uuid():
    """
    生成标准 UUID 字符串
    
    Returns:
        str: UUID 字符串
    """
    return str(uuid.uuid4())


def vector_id(set_id, content):
    """
    为向量生成唯一标识符
    
    基于 set_id 和 content 内容生成哈希值，确保同样内容生成相同的 ID
    
    Args:
        set_id (str): 认元集合 ID
        content (str): 认元内容
        
    Returns:
        str: 生成的哈希 ID
    """
    # 确保输入是字符串类型
    set_id_str = str(set_id)
    content_str = str(content)
    
    # 将 set_id 和 content 组合起来
    combined = f"{set_id_str}:{content_str}"
    
    # 计算 SHA-256 哈希值
    hash_obj = hashlib.sha256(combined.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    # 取前 32 位作为 ID
    return hash_hex[:32]


def generate_content_hash_id(set_id, content):
    """
    基于 set_id 和 content 生成哈希 ID
    
    使用 SHA-256 算法对内容进行哈希，然后取前 32 位作为 ID
    注意：如果内容完全相同，生成的 ID 也会相同
    
    Args:
        set_id (str): 认元集合 ID
        content (str): 认元内容
        
    Returns:
        str: 生成的哈希 ID
    """
    return vector_id(set_id, content)


def generate_unique_content_hash_id(set_id, content):
    """
    基于 set_id 和 content 生成唯一哈希 ID
    
    与 generate_content_hash_id 不同，这个函数总是生成唯一 ID，
    即使内容相同也会生成不同的 ID
    
    Args:
        set_id (str): 认元集合 ID
        content (str): 认元内容
        
    Returns:
        str: 生成的唯一哈希 ID
    """
    # 添加一个时间戳和随机 UUID 确保唯一性
    random_part = str(uuid.uuid4())
    
    # 确保输入是字符串类型
    set_id_str = str(set_id)
    content_str = str(content)
    
    # 将 set_id、content 和随机部分组合起来
    combined = f"{set_id_str}:{content_str}:{random_part}"
    
    # 计算 SHA-256 哈希值
    hash_obj = hashlib.sha256(combined.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    # 取前 32 位作为 ID
    return hash_hex[:32] 