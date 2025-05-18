# 记忆激活模型（Memory Activation Model）

MAM 是一个专注于计算认元权重的工具模块，根据多种因素确定认元在认知网络中的重要性和激活程度。

## 主要功能

- 基于时间因子的权重计算
- 基于连接数量的权重计算
- 基于访问频率的权重计算
- 基于认知特性（重要性、情感强度、用户标记）的权重计算
- 综合多种因素计算最终权重

## 技术原理

MAM 使用多种因素综合评估认元的权重，包括旧权重、时间因子、连接数量、访问频率和认知特性等。通过加权组合这些因素生成最终权重。

```
认元权重 = f(旧权重, 时间, 连接, 频率, 认知特性)
```

## 主要接口

### 认元权重计算

```python
def coglet_weight(old_weight: float = 0.5, 
                  time_factor: float = 1.0,
                  connection_count: int = 0, 
                  access_count: int = 0,
                  importance: float = 0.5,
                  emotion_intensity: float = 0.0,
                  user_marked: bool = False) -> float:
```

计算认元的综合权重。

#### 参数

- `old_weight`: 认元的旧权重值，默认为0.5
- `time_factor`: 时间因子，表示时间衰减的程度，范围0-1，默认为1.0（无衰减）
- `connection_count`: 认元连接数，默认为0
- `access_count`: 认元访问次数，默认为0
- `importance`: 认元重要性，范围0-1，默认为0.5
- `emotion_intensity`: 情感强度，范围0-1，默认为0
- `user_marked`: 是否被用户标记，默认为False

#### 返回值

- 认元最终权重，范围0.1-1.0

## 权重计算模型

### 内部计算逻辑

该函数内部采用以下步骤计算最终权重：

1. 时间因子直接用作时间权重
2. 连接权重使用 Sigmoid 函数计算
3. 频率权重使用对数函数计算
4. 认知权重基于重要性、情感强度和用户标记计算
5. 通过加权平均组合所有权重因素

### 权重分配

默认的权重分配：
- 旧权重：30%
- 时间因子：20%
- 连接数量：15%
- 访问频率：15%
- 认知特性：20%

## 使用场景

1. **认元激活计算**：决定哪些认元应该在网络传播中被激活
2. **认元排序**：对搜索结果进行排序
3. **记忆清理**：决定哪些低权重认元可以被清理
4. **认知焦点**：确定网络中的关注焦点

## 示例

### 基本使用

```python
from src.utils.mam import coglet_weight

# 计算基本权重
weight = coglet_weight()
print(f"默认权重: {weight:.4f}")  # 0.5000

# 计算高权重认元
high_weight = coglet_weight(
    old_weight=0.6,
    time_factor=0.9,
    connection_count=15,
    access_count=50,
    importance=0.8,
    emotion_intensity=0.7,
    user_marked=True
)
print(f"高权重认元: {high_weight:.4f}")  # 约 0.7500
```

### 实际应用

```python
# 示例：根据认元的使用情况更新权重
def update_coglet_weight(coglet_data):
    # 获取当前权重
    current_weight = coglet_data.get("weight", 0.5)
    
    # 计算时间因子（假设根据最后访问时间计算）
    last_access = coglet_data.get("last_access_time", 0)
    hours_passed = (time.time() - last_access) / 3600
    time_factor = math.exp(-0.05 * hours_passed)  # 指数衰减
    
    # 计算新权重
    new_weight = coglet_weight(
        old_weight=current_weight,
        time_factor=time_factor,
        connection_count=coglet_data.get("connection_count", 0),
        access_count=coglet_data.get("access_count", 0),
        importance=coglet_data.get("importance", 0.5),
        emotion_intensity=coglet_data.get("emotion_intensity", 0),
        user_marked=coglet_data.get("user_marked", False)
    )
    
    return new_weight
```

# 记忆锚定机制（Memory Anchor Mechanism）

MAM 是一个提供认元权重计算和记忆管理功能的模块，根据时间因素和交互频率动态调整认元在认知网络中的重要性和激活程度。

## 主要功能

- 基于公式 W_{t+1} = e^{-b·Δt} · (W_t·β + γ·Δt) 计算认元权重
- 创建、获取、更新和删除记忆集合
- 添加、获取、更新和删除记忆
- 基于相似度的记忆检索（回忆）功能，支持黄金分割比例激活
- 权重随时间的动态更新
- 支持批量刷新记忆权重

## 技术原理

MAM 使用带有时间衰减的权重计算公式来模拟人类记忆的间隔重复效应：

```
W_{t+1} = e^{-b·Δt} · (W_t·β + γ·Δt)
```

其中：
- W_{t+1} 是更新后的权重
- W_t 是当前权重
- Δt 是时间间隔（小时）
- β (beta) 是旧记忆残留因子（默认0.85）
- γ (gamma) 是新调用强化增益系数（默认0.3）
- b 是时间敏感系数，表示遗忘速率（默认0.05）

### 黄金分割激活

回忆（recall）功能使用黄金分割比例（0.618）来选择权重排序后前61.8%的记忆进行激活，这种方式符合自然认知规律，使得激活模式更加自然和高效。

## 初始化参数

```python
def __init__(
    self, 
    vector_store=None,
    beta: float = 0.85, 
    gamma: float = 0.3, 
    b: float = 0.05,
    initial_weight: float = 0.5,
    golden_ratio: float = 0.618,  # 黄金分割比例
    recall_top_k: int = 10        # 回忆检索的默认数量
):
```

关键参数说明：
- `beta`: 旧记忆残留因子，控制历史记忆的保留比例
- `gamma`: 新调用强化增益系数，控制每次调用的增益
- `b`: 时间敏感系数(遗忘速率)，控制记忆随时间的衰减速度
- `initial_weight`: 首次创建认元的初始权重
- `golden_ratio`: 黄金分割比例(0.618)，用于在recall中选择激活记忆的阈值
- `recall_top_k`: 回忆检索的默认数量

## 主要接口

### 认元权重计算

```python
def coglet_weight(
    self,
    last_weight: Optional[float] = None, 
    last_time: Optional[float] = None,
    current_time: Optional[float] = None,
    beta: Optional[float] = None, 
    gamma: Optional[float] = None, 
    b: Optional[float] = None
) -> float:
```

计算认元的记忆权重更新。

#### 参数

- `last_weight`: 上一次记忆强度，如果为None表示首次创建认元
- `last_time`: 上次更新时间（秒），如果为None表示首次创建认元
- `current_time`: 当前时间（秒），默认为当前系统时间
- `beta`: 旧记忆残留因子，如果为None则使用实例的默认值
- `gamma`: 新调用强化增益系数，如果为None则使用实例的默认值
- `b`: 时间敏感系数，如果为None则使用实例的默认值

#### 返回值

- 更新后的记忆强度，范围0.0-1.0

### 集合管理

```python
def create_memory_set(self, set_id: str, description: Optional[str] = None) -> bool:
def get_set_info(self, set_id: str) -> Dict[str, Any]:
def list_sets(self) -> List[Dict[str, Any]]:
def delete_set(self, set_id: str) -> bool:
```

创建、获取、列出和删除记忆集合。

### 记忆管理

```python
def add(self, set_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
def get(self, memory_id: str) -> Dict[str, Any]:
def update(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
def refresh(self, memory_ids: Union[str, List[str]]) -> List[str]:
def delete(self, memory_id: str) -> bool:
```

添加、获取、更新、刷新和删除记忆。

特别说明：
- `refresh` 方法支持单个ID或ID列表，返回成功刷新的ID列表

### 记忆检索

```python
def recall(self, set_id: str, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
```

基于相似度检索记忆，并使用黄金分割比例选择激活记忆。

#### 参数

- `set_id`: 集合ID
- `query`: 查询内容
- `top_k`: 返回的最大记忆数量，默认使用初始化时设定的值

#### 返回值

返回包含所有结果和激活记忆的字典:
```python
{
    "all_results": [...],  # 所有检索结果（按权重排序）
    "activated": [...]     # 被激活的记忆（权重排序后前golden_ratio部分）
}
```

## 使用示例

```python
# 初始化MAM
from src.utils.mam import MAM
from src.core.vector_store import VectorStore

vector_store = VectorStore(url, token)
# 使用自定义参数初始化
mam = MAM(
    vector_store=vector_store,
    beta=0.9,                # 更高的旧记忆残留
    gamma=0.2,               # 更低的新调用增益
    b=0.03,                  # 更慢的遗忘速率
    golden_ratio=0.5,        # 只激活前50%的记忆
    recall_top_k=20          # 默认检索20条记忆
)

# 创建记忆集合
mam.create_memory_set("my_memories", "我的重要记忆")

# 添加记忆
memory_ids = []
for i in range(5):
    memory_id = mam.add("my_memories", f"记忆内容示例 {i}", {"importance": i})
    memory_ids.append(memory_id)

# 批量刷新记忆
refreshed_ids = mam.refresh(memory_ids)
print(f"成功刷新的记忆数: {len(refreshed_ids)}")

# 回忆相关内容并使用黄金分割激活
results = mam.recall("my_memories", "查询内容")
print(f"找到 {len(results['all_results'])} 条记忆")
print(f"其中 {len(results['activated'])} 条被激活")

# 处理激活的记忆
for memory in results['activated']:
    print(f"激活记忆: {memory['content']}, 权重: {memory['metadata']['weight']}")
```

## 权重动态变化

MAM模型中的权重计算考虑了时间因素，呈现以下特性：

1. 随时间自然衰减（遗忘曲线）
2. 每次访问/激活会增强权重
3. 在最佳间隔时间复习效果更佳
4. 高频访问的记忆权重更高

此外，使用黄金分割比例选择激活记忆，能够在保持认知网络高效运行的同时，确保关注点集中在最重要的记忆上。
 