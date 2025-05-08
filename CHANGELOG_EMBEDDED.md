# 变更日志：Upstash Vector 内置嵌入功能集成

## 概述

本次更新将 Upstash Vector 存储服务从依赖 OpenAI API 生成嵌入向量迁移到使用 Upstash Vector 的内置嵌入功能。这一改进简化了架构，降低了成本，并减少了对外部服务的依赖。

## 主要变更

### 1. 代码优化

- **移除了 OpenAI API 依赖**：不再需要 `OPENAI_API_KEY` 环境变量
- **移除了 Redis 依赖**：不再需要 `UPSTASH_REDIS_URL` 和 `UPSTASH_REDIS_TOKEN` 环境变量
- **简化了 `UpstashVectorStore` 类**：
  - 移除了 `_get_embedding` 方法，不再需要手动生成嵌入向量
  - 删除了与 OpenAI 相关的配置和处理逻辑
  - 移除了不必要的方法和参数
- **更简洁的 API 设计**：
  - 直接使用文本内容而非嵌入向量进行存储和查询
  - 提供更清晰的错误信息和日志记录

### 2. 新增功能

- **内置嵌入模型集成**：
  - 默认使用 `BAAI/bge-small-en-v1.5` 嵌入模型
  - 支持配置不同的嵌入模型
- **直接文本查询**：
  - 新增 `query` 方法支持直接文本查询
  - 支持元数据过滤和排序

### 3. 性能改进

- **减少 API 调用**：从两次 API 调用（OpenAI + Upstash）减少到一次 API 调用（仅 Upstash）
- **降低延迟**：消除了通过 OpenAI API 获取嵌入的延迟
- **提高可靠性**：降低了因依赖多个外部服务导致的故障率

### 4. 新增工具和文档

- **新增测试工具**：
  - `upstash_vector_demo.py`：演示 Upstash Vector 内置嵌入功能的使用方法
  - `check_server_embedded.py`：检查服务器状态和索引信息
- **详细文档**：
  - `UPSTASH_VECTOR_README.md`：使用指南和最佳实践
  - `CHANGELOG_EMBEDDED.md`：变更历史记录

## 实现细节

### 主要代码修改

1. **初始化方法**：
```python
def __init__(
    self,
    vector_url: Optional[str] = None,
    vector_token: Optional[str] = None,
    embedding_model: str = "BAAI/bge-small-en-v1.5"
):
    # 使用 Upstash 的内置嵌入功能
    self.index = Index(
        url=self.vector_url,
        token=self.vector_token
    )
```

2. **添加认元**：
```python
async def add_coglet(self, content: str, ...):
    # 直接使用文本内容
    self.index.upsert([
        {
            "id": coglet_id,
            "data": content,  # 直接使用文本内容
            "metadata": coglet_metadata
        }
    ])
```

3. **搜索相似认元**：
```python
async def search_similar(self, query: str, ...):
    query_params = {
        "data": query,  # 直接使用文本查询
        "top_k": top_k,
        "include_metadata": True
    }
```

## 兼容性

- 这些更改与现有的代码是向后兼容的
- 已有的数据不会受到影响，但新数据将使用内置嵌入功能存储

## 下一步计划

- 研究并测试更多嵌入模型的性能和准确性
- 探索 Upstash Vector 的其他高级功能，如混合索引和稀疏向量
- 实现更完善的命名空间管理和数据组织方式 