#!/usr/bin/env python
"""
检查 Upstash Vector 服务器状态（使用内置嵌入模型）
"""
import asyncio
import os
import sys
from dotenv import load_dotenv
from upstash_vector.errors import UpstashError
from src.utils.vector_store import UpstashVectorStore, check_environment
from src.utils.prompts_config import VECTOR_STORE_CONFIG

async def check_vector_service(store):
    """详细检查向量服务的各个方面
    
    Args:
        store: UpstashVectorStore 实例
        
    Returns:
        (检查结果, 详细信息)
    """
    print("\n详细健康检查:")
    
    # 检查1: 基本连接
    print("1. 检查基本连接...")
    if not store.index:
        return False, "Upstash 客户端未初始化"
        
    if not store.vector_url or not store.vector_token:
        return False, "Upstash 凭证未正确配置"
    
    # 检查2: 查询功能
    print("2. 测试查询功能...")
    try:
        test_query = VECTOR_STORE_CONFIG["HEALTH_CHECK_TEXT"]
        print(f"   执行测试查询: '{test_query}'")
        
        results = store.index.query(
            data=test_query,
            top_k=1,
            include_metadata=True
        )
        
        if not isinstance(results, list):
            return False, f"查询结果类型错误: 期望列表, 实际为 {type(results)}"
            
        print(f"   ✅ 查询功能正常")
    except UpstashError as e:
        error_msg = f"Upstash 服务器错误: {str(e)}"
        print(f"   ❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"查询测试失败: {str(e)}"
        print(f"   ❌ {error_msg}")
        return False, error_msg
    
    # 检查3: 测试多语言嵌入支持...
    print("3. 测试多语言嵌入支持...")
    try:
        multi_lang_texts = [
            ("英文", "This is an English text for testing embedding"),
            ("中文", "这是一段用于测试嵌入的中文文本"),
            ("混合", "This is a mixed text with English and 中文内容")
        ]
        
        for lang, text in multi_lang_texts:
            print(f"   测试{lang}文本嵌入: '{text[:20]}...'")
            results = store.index.query(
                data=text,
                top_k=1,
                include_metadata=True
            )
            
            if not isinstance(results, list):
                return False, f"{lang}文本嵌入测试失败: 查询结果类型错误"
                
            print(f"   ✅ {lang}文本嵌入测试通过")
    except Exception as e:
        error_msg = f"多语言嵌入测试失败: {str(e)}"
        print(f"   ❌ {error_msg}")
        return False, error_msg
    
    # 检查4: 测试添加和检索功能
    print("4. 测试添加和检索功能...")
    try:
        # 创建唯一的测试集合ID
        test_collection = f"health_check_{int(asyncio.get_event_loop().time())}"
        test_content = "这是一个健康检查测试内容"
        test_timestamp = asyncio.get_event_loop().time()
        
        print(f"   添加测试数据到集合 '{test_collection}'...")
        coglet_id = await store.add_coglet(
            content=test_content,
            weight=1.0,
            timestamp=test_timestamp,
            collection_id=test_collection
        )
        
        print(f"   ✅ 添加成功, ID: {coglet_id}")
        
        # 等待索引更新
        print(f"   等待索引更新...")
        await asyncio.sleep(1)
        
        # 查询添加的内容
        print(f"   检索添加的内容...")
        results = await store.search_similar(
            query=test_content[:20],  # 使用部分内容作为查询
            collection_id=test_collection,
            top_k=1
        )
        
        if not results or len(results) == 0:
            return False, "无法检索到添加的测试数据"
            
        print(f"   ✅ 检索成功, 找到 {len(results)} 个结果")
        print(f"   ✅ 相似度分数: {results[0][2]:.4f}")
        
        # 测试使用不同的 top_k 值
        print(f"   测试不同的 top_k 值...")
        for top_k in [1, 3, 5]:
            results = await store.search_similar(
                query=test_content[:20],
                collection_id=test_collection,
                top_k=top_k
            )
            print(f"   ✅ top_k={top_k}: 找到 {len(results)} 个结果")
        
        # 测试使用不同的 min_score 值
        print(f"   测试不同的 min_score 值...")
        for min_score in [0.5, 0.7, 0.9]:
            results = await store.search_similar(
                query=test_content[:20],
                collection_id=test_collection,
                top_k=5,
                min_score=min_score
            )
            print(f"   ✅ min_score={min_score}: 找到 {len(results)} 个结果")
        
        # 测试直接查询方法
        print(f"   测试直接查询方法...")
        query_results = await store.query(
            text=test_content[:20],
            top_k=3,
            filter=f"collection_id = '{test_collection}'"
        )
        
        if not query_results or len(query_results) == 0:
            return False, "使用直接查询方法无法检索到测试数据"
            
        print(f"   ✅ 直接查询成功, 找到 {len(query_results)} 个结果")
        print(f"   ✅ 第一个结果ID: {query_results[0]['id']}")
        print(f"   ✅ 相似度分数: {query_results[0]['score']:.4f}")
        
        # 清理测试数据
        print(f"   清理测试数据...")
        store.index.delete([coglet_id])
        print(f"   ✅ 测试数据已清理")
        
    except Exception as e:
        error_msg = f"添加和检索测试失败: {str(e)}"
        print(f"   ❌ {error_msg}")
        return False, error_msg
    
    # 检查5: 测试错误处理和边缘情况
    print("5. 测试错误处理和边缘情况...")
    try:
        # 测试空查询
        print(f"   测试空查询...")
        empty_results = await store.search_similar(
            query="",
            top_k=1
        )
        print(f"   ✅ 空查询测试通过")
        
        # 测试超长查询
        print(f"   测试超长查询...")
        long_text = "测试文本 " * 100  # 创建一个长文本
        long_results = await store.search_similar(
            query=long_text,
            top_k=1
        )
        print(f"   ✅ 超长查询测试通过")
        
        # 测试特殊字符
        print(f"   测试包含特殊字符的查询...")
        special_chars = "测试!@#$%^&*()_+{}:|<>?~`-=[]\\;',./查询"
        special_results = await store.search_similar(
            query=special_chars,
            top_k=1
        )
        print(f"   ✅ 特殊字符查询测试通过")
        
    except Exception as e:
        # 这里我们只记录错误但不算作健康检查失败
        error_msg = f"边缘情况测试出现异常: {str(e)}"
        print(f"   ⚠️ {error_msg}")
        print(f"   ⚠️ 这些是边缘测试，不会导致健康检查失败")
    
    return True, "所有检查通过"

async def main():
    """检查服务器状态"""
    print("Upstash Vector 服务器状态检查（使用内置嵌入功能）")
    print("================================================")
    
    # 加载环境变量
    load_dotenv()
    
    # 检查环境变量
    all_set, missing_vars = check_environment()
    
    # 打印环境变量状态
    print("\n环境变量检查:")
    if all_set:
        print("✅ 所有必要的环境变量已设置")
        # 打印变量值（隐藏敏感信息）
        vector_url = os.getenv("UPSTASH_VECTOR_URL")
        vector_token = os.getenv("UPSTASH_VECTOR_TOKEN")
        print(f"UPSTASH_VECTOR_URL: {vector_url[:20]}..." if vector_url else "")
        print(f"UPSTASH_VECTOR_TOKEN: {vector_token[:10]}..." if vector_token else "")
    else:
        print("❌ 缺少必要的环境变量:")
        for var in missing_vars:
            print(f"   - {var}")
        sys.exit(1)
    
    # 打印 Upstash Vector 信息
    print("\n初始化 Upstash Vector 客户端")
    try:
        # 创建 UpstashVectorStore 实例（使用内置嵌入功能）
        store = UpstashVectorStore(
            embedding_model=VECTOR_STORE_CONFIG["DEFAULT_EMBEDDING_MODEL"]  # 显式指定使用的嵌入模型
        )
        
        # 基本健康检查
        print("\n执行基本健康检查...")
        healthy, message = await store.check_health()
        
        # 打印结果
        if healthy:
            print(f"✅ 基本健康检查: 通过")
            print(f"✅ 状态信息: {message}")
            
            # 执行详细健康检查
            detail_healthy, detail_message = await check_vector_service(store)
            
            if detail_healthy:
                print(f"\n✅ 详细健康检查: 通过")
                print(f"✅ 详细信息: {detail_message}")
            else:
                print(f"\n❌ 详细健康检查: 失败")
                print(f"❌ 失败原因: {detail_message}")
                sys.exit(1)
            
            # 获取关于索引的信息
            print("\n获取索引信息...")
            info = store.index.info()
            print("✅ 索引信息:")
            
            # 正确处理 InfoResult 对象
            if hasattr(info, 'vector_count'):
                print(f"  - 向量数量: {info.vector_count}")
                print(f"  - 待处理向量数量: {info.pending_vector_count}")
                print(f"  - 索引大小: {info.index_size} 字节")
                print(f"  - 向量维度: {info.dimension}")
                print(f"  - 相似度函数: {info.similarity_function}")
            else:
                # 尝试使用变通方法，将对象打印出来
                print(f"  索引信息详情: {info}")
                print(f"  索引信息类型: {type(info)}")
                print(f"  索引信息属性: {dir(info)}")
            
            print("\n所有检查通过 ✅")
        else:
            print(f"❌ 基本健康检查: 失败")
            print(f"❌ 失败原因: {message}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ 出现错误: {str(e)}")
        print(f"❌ 错误类型: {type(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
    
if __name__ == "__main__":
    asyncio.run(main()) 