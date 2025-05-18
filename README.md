# CogletNet

一个基于 Upstash Vector 的向量存储实现。

## 功能特性

- 支持认元的添加、获取、更新和删除
- 支持批量操作
- 支持相似度搜索
- 支持认元集合管理

## 安装

```bash
pip install -e ".[dev]"
```

## 使用示例

```python
from src.core.vector_store import VectorStore

# 初始化向量存储
store = VectorStore(
    url="your-upstash-url",
    token="your-upstash-token"
)

# 添加认元
coglet_id = store.add_coglet(
    set_id="my-set",
    content="这是一个测试认元",
    metadata={"type": "test"}
)

# 搜索相似认元
results = store.search_similar(
    set_id="my-set",
    query="测试认元",
    top_k=5
)
```

## 开发

1. 克隆仓库
2. 创建虚拟环境：`python -m venv venv`
3. 激活虚拟环境：`source venv/bin/activate`
4. 安装开发依赖：`pip install -e ".[dev]"`
5. 运行测试：`pytest tests/` 