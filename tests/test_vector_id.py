"""
测试 vector_id 函数
"""

from src.id_generator import vector_id, generate_uuid

# 测试相同内容生成相同 ID
set_id1 = "test_set"
content1 = "这是一条测试内容"

id1 = vector_id(set_id1, content1)
id2 = vector_id(set_id1, content1)

print(f"对于相同内容生成的 ID:")
print(f"ID1: {id1}")
print(f"ID2: {id2}")
print(f"ID1 == ID2: {id1 == id2}")
print()

# 测试不同内容生成不同 ID
content2 = "这是另一条测试内容"
id3 = vector_id(set_id1, content2)

print(f"对于不同内容生成的 ID:")
print(f"ID1: {id1}")
print(f"ID3: {id3}")
print(f"ID1 == ID3: {id1 == id3}")
print()

# 测试不同集合 ID
set_id2 = "another_set"
id4 = vector_id(set_id2, content1)

print(f"对于相同内容但不同集合的 ID:")
print(f"ID1 (set_id={set_id1}): {id1}")
print(f"ID4 (set_id={set_id2}): {id4}")
print(f"ID1 == ID4: {id1 == id4}")
print()

# 与 UUID 比较
uuid1 = generate_uuid()
uuid2 = generate_uuid()

print(f"UUID 示例:")
print(f"UUID1: {uuid1}")
print(f"UUID2: {uuid2}")
print(f"UUID1 == UUID2: {uuid1 == uuid2}")
print()

# 显示向量 ID 的格式
print(f"vector_id 格式示例: {id1}")
print(f"长度: {len(id1)} 位")
print(f"字符类型: 十六进制") 