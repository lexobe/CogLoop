# ID 生成工具

ID 生成工具提供了多种生成唯一标识符的方法，主要用于认元（Coglet）ID生成。

## 主要功能

- 基于内容的确定性ID生成
- UUID生成
- 混合ID生成（基于内容但确保唯一）

## 函数说明

### vector_id

```python
def vector_id(set_id: str, content: str) -> str:
```

这是主要的ID生成函数，基于集合ID和内容生成一个确定性的32位哈希ID。对于相同的输入，总是生成相同的ID。

#### 参数
- `set_id`: 认元集合ID
- `content`: 认元内容

#### 返回值
- 32位十六进制字符串

#### 示例
```python
from src.utils.id_generator import vector_id

# 生成ID
coglet_id = vector_id("test_set", "这是认元内容")
print(coglet_id)  # 例如: caf0a210120e96459540c088e8b95aad
```

### generate_uuid

```python
def generate_uuid() -> str:
```

生成一个标准UUID（通用唯一标识符）字符串。每次调用都会生成一个全新的、统计上唯一的ID。

#### 返回值
- UUID字符串

#### 示例
```python
from src.utils.id_generator import generate_uuid

# 生成UUID
uuid_str = generate_uuid()
print(uuid_str)  # 例如: 424ff87a-f2da-4fbb-9e44-9197ad5afd4d
```

### generate_content_hash_id

```python
def generate_content_hash_id(set_id: str, content: str) -> str:
```

基于集合ID和内容生成哈希ID。这个函数是 `vector_id` 的别名，保留用于向后兼容性。

### generate_unique_content_hash_id

```python
def generate_unique_content_hash_id(set_id: str, content: str) -> str:
```

生成一个基于内容但始终唯一的ID。即使输入内容完全相同，每次调用也会生成一个不同的ID。

#### 参数
- `set_id`: 认元集合ID
- `content`: 认元内容

#### 返回值
- 32位十六进制字符串

#### 示例
```python
from src.utils.id_generator import generate_unique_content_hash_id

# 生成唯一ID
unique_id1 = generate_unique_content_hash_id("test_set", "内容")
unique_id2 = generate_unique_content_hash_id("test_set", "内容")
print(unique_id1 != unique_id2)  # True，两个ID永远不同
```

## 使用场景

1. **确定性ID**：当需要为相同内容生成相同ID时，使用 `vector_id`。适用于:
   - 去重场景
   - 内容寻址
   - 缓存键生成

2. **非确定性ID**：当需要保证ID唯一性时，使用 `generate_uuid`。适用于:
   - 新实体创建
   - 随机标识符
   - 临时ID

3. **混合ID**：当需要ID包含内容信息但又要确保唯一性时，使用 `generate_unique_content_hash_id` 