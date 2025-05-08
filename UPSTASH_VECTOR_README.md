# Upstash Vector 内置嵌入功能使用指南

## 概述

本文档介绍了如何使用 Upstash Vector 的内置嵌入功能，无需依赖第三方嵌入服务（如 OpenAI）。通过使用 Upstash Vector 的内置嵌入功能，可以简化开发流程，降低成本，并减少对外部服务的依赖。

## 优势

1. **简化架构**：不需要额外的嵌入服务，减少了系统复杂性
2. **降低成本**：避免了使用 OpenAI API 的费用
3. **减少延迟**：少了一次外部 API 调用，系统响应更快
4. **提高可靠性**：消除了对外部嵌入服务的依赖，提高了系统的可用性
5. **简化部署**：无需管理额外的 API 密钥和配置
6. **移除 Redis 依赖**：不再需要配置和维护 Redis 服务

## 配置方法

### 1. 创建 Upstash Vector 数据库

1. 前往 [Upstash Console](https://console.upstash.com/) 注册/登录
2. 创建一个新的 Vector 数据库
3. 在创建过程中选择一个嵌入模型（如 `BAAI/bge-small-en-v1.5`）
4. 记录数据库的 URL 和 Token

### 2. 环境变量设置

将以下环境变量添加到 `.env` 文件中：

```
UPSTASH_VECTOR_URL=https://your-upstash-vector-url.upstash.io
UPSTASH_VECTOR_TOKEN=your-upstash-vector-token
```

如果使用内置嵌入功能，无需设置 `OPENAI_API_KEY`。

### 3. 代码集成

`UpstashVectorStore` 类已经更新为支持 Upstash Vector 的内置嵌入功能。初始化时无需提供嵌入模型实例：

```python
from src.utils.vector_store import UpstashVectorStore

# 使用 Upstash Vector 的内置嵌入功能
store = UpstashVectorStore()

# 也可以指定嵌入模型
# store = UpstashVectorStore(embedding_model="BAAI/bge-large-en-v1.5")
```

## 使用方法

### 添加认元

直接使用文本内容添加认元，无需手动生成嵌入向量：

```python
coglet_id = await store.add_coglet(
    content="这是一段要存储的文本内容",
    weight=1.0,
    timestamp=1625097600.0,
    collection_id="my_collection"
)
```

### 搜索相似认元

使用文本直接搜索相似认元：

```python
results = await store.search_similar(
    query="查询文本",
    collection_id="my_collection",
    top_k=5,
    min_score=0.7
)

# 处理结果
for doc_id, metadata, score in results:
    print(f"ID: {doc_id}, 内容: {metadata['content']}, 相似度: {score}")
```

### 直接查询

使用 `query` 方法可以直接进行文本查询：

```python
query_results = await store.query(
    text="查询文本",
    top_k=3,
    filter="collection_id = 'my_collection'"
)

# 处理结果
for result in query_results:
    print(f"ID: {result['id']}, 内容: {result['metadata']['content']}, 相似度: {result['score']}")
```

## 支持的嵌入模型

Upstash Vector 支持多种嵌入模型，选择模型时请考虑您的具体需求：

| 模型名称 | 维度 | 特点 |
|---------|------|------|
| BAAI/bge-small-en-v1.5 | 384 | 小型模型，低延迟，适合一般用途 |
| BAAI/bge-base-en-v1.5 | 768 | 平衡大小和性能的中型模型 |
| BAAI/bge-large-en-v1.5 | 1024 | 大型模型，性能最佳但延迟较高 |
| WhereIsAI/UAE-Large-V1 | 1024 | 高精度通用嵌入模型 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 轻量级模型，适合资源受限场景 |

更多信息请参考 [Upstash Vector 文档](https://upstash.com/docs/vector/features/embeddingmodels)。

## 示例代码

完整的示例代码请参考 `upstash_vector_demo.py`，该脚本演示了如何使用 Upstash Vector 的内置嵌入功能进行数据存储和查询。

## 故障排除

如果遇到问题，请检查以下几点：

1. 确保环境变量 `UPSTASH_VECTOR_URL` 和 `UPSTASH_VECTOR_TOKEN` 已正确设置
2. 检查 Upstash Vector 数据库是否已配置嵌入模型
3. 使用 `check_health()` 方法验证服务器连接是否正常
4. 检查日志输出，了解详细的错误信息

## 注意事项

1. 添加数据后，索引需要一些时间来更新。在高吞吐量场景下，可能需要等待几秒钟后再进行查询。
2. 不同的嵌入模型可能会产生不同的查询结果，请根据您的应用场景选择适当的模型。
3. 文本内容的质量和相关性对查询结果有显著影响。 