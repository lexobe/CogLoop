#!/usr/bin/env python
"""
Upstash Vector 内置嵌入功能演示

此脚本演示如何使用 Upstash Vector 的内置嵌入功能，
无需使用第三方嵌入服务如 OpenAI。
"""
import asyncio
import os
from dotenv import load_dotenv
from src.utils.vector_store import UpstashVectorStore, check_environment

async def demo():
    """演示 Upstash Vector 内置嵌入功能"""
    print("Upstash Vector 内置嵌入功能演示")
    print("===================================")
    
    # 加载环境变量
    load_dotenv()
    
    # 检查环境变量
    all_set, missing_vars = check_environment()
    if not all_set:
        print("❌ 缺少必要的环境变量:")
        for var in missing_vars:
            print(f"   - {var}")
        return
        
    print("✅ 环境变量设置正确")
    
    # 创建 UpstashVectorStore 实例
    print("\n初始化 UpstashVectorStore...")
    # 默认使用 "BAAI/bge-small-en-v1.5" 嵌入模型
    store = UpstashVectorStore()
    
    # 检查服务器健康状态
    print("\n检查服务器健康状态...")
    healthy, message = await store.check_health()
    print(f"服务器状态: {'✅ 正常' if healthy else '❌ 异常'}")
    print(f"状态信息: {message}")
    
    if not healthy:
        return
        
    # 存储示例数据
    print("\n存储示例数据...")
    collection_id = "demo_collection"
    
    # 存储多个示例数据
    sample_data = [
        "Python 是一种强类型、动态类型的编程语言，支持面向对象、命令式、函数式和过程式编程范式。",
        "TensorFlow 是一个由 Google 开发的开源机器学习框架，用于构建和训练神经网络模型。",
        "PyTorch 是 Facebook 的 AI 研究实验室开发的开源机器学习库，基于 Torch 库。",
        "NumPy 是 Python 编程语言的一个扩展程序库，支持大量的维度数组与矩阵运算。",
        "Pandas 是用于数据分析和数据操作的 Python 库，提供了高性能、易用的数据结构和数据分析工具。"
    ]
    
    ids = []
    for i, content in enumerate(sample_data):
        timestamp = float(1625097600 + i * 3600)  # 每条数据间隔1小时
        coglet_id = await store.add_coglet(
            content=content,
            weight=1.0,
            timestamp=timestamp,
            collection_id=collection_id
        )
        ids.append(coglet_id)
        print(f"✅ 已添加数据: ID={coglet_id}")
    
    # 等待索引更新
    print("\n等待索引更新...")
    await asyncio.sleep(2)
    
    # 执行搜索查询
    print("\n执行搜索查询...")
    queries = [
        "Python 编程语言特性",
        "机器学习框架",
        "数据分析工具"
    ]
    
    for query in queries:
        print(f"\n查询: '{query}'")
        results = await store.search_similar(
            query=query,
            collection_id=collection_id,
            top_k=3
        )
        
        print(f"找到 {len(results)} 个结果:")
        for idx, (doc_id, metadata, score) in enumerate(results):
            print(f"结果 {idx+1}:")
            print(f"  ID: {doc_id}")
            print(f"  内容: {metadata['content'][:60]}...")
            print(f"  相似度: {score:.4f}")
    
    # 使用 query 方法
    print("\n使用 query 方法直接查询...")
    query_text = "机器学习工具"
    print(f"查询: '{query_text}'")
    
    query_results = await store.query(
        text=query_text,
        top_k=3,
        filter=f"collection_id = '{collection_id}'"
    )
    
    print(f"找到 {len(query_results)} 个结果:")
    for idx, result in enumerate(query_results):
        print(f"结果 {idx+1}:")
        print(f"  ID: {result['id']}")
        print(f"  内容: {result['metadata']['content'][:60]}...")
        print(f"  相似度: {result['score']:.4f}")
    
    print("\n演示完成!")

if __name__ == "__main__":
    asyncio.run(demo()) 