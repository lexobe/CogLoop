#!/usr/bin/env python
"""
检查 LLM 服务器状态（支持多种模型服务）
"""
import asyncio
import os
import sys
import time
import json
from dotenv import load_dotenv
import litellm
from tenacity import retry, stop_after_attempt, wait_fixed

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 测试提示词
TEST_PROMPTS = {
    "基本验证": "这是一个测试消息，请回复'LLM服务器正常运行'",
    "中文能力": "请用一句话描述人工智能的未来",
    "代码能力": "写一个简单的Python函数计算斐波那契数列的第n项",
    "推理能力": "如果所有的猫都有四条腿，小花是一只猫，那么小花有几条腿？请解释你的推理过程。"
}

# 支持的LLM提供商及其默认模型
SUPPORTED_PROVIDERS = {
    "openai": ["gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    "deepseek": ["deepseek-chat"],
    "gemini": ["gemini-pro"]
}

async def check_environment() -> tuple[bool, list]:
    """检查必要的环境变量是否已设置"""
    # 加载环境变量
    load_dotenv()
    
    required_vars = {}
    
    # 检查OpenAI API密钥
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        required_vars["OPENAI_API_KEY"] = True
    else:
        required_vars["OPENAI_API_KEY"] = False
    
    # 检查Anthropic API密钥
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        required_vars["ANTHROPIC_API_KEY"] = True
    else:
        required_vars["ANTHROPIC_API_KEY"] = False
    
    # 检查DeepSeek API密钥
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        required_vars["DEEPSEEK_API_KEY"] = True
    else:
        required_vars["DEEPSEEK_API_KEY"] = False
    
    # 检查Gemini API密钥
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        required_vars["GEMINI_API_KEY"] = True
    else:
        required_vars["GEMINI_API_KEY"] = False
    
    missing_vars = [var for var, is_set in required_vars.items() if not is_set]
    
    # 如果至少有一个API密钥设置，则认为环境变量足够
    all_set = any(required_vars.values())
    
    return all_set, missing_vars

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def test_llm_request(provider, model, prompt):
    """测试LLM请求，带有重试机制
    
    Args:
        provider: 服务提供商名称
        model: 模型名称
        prompt: 测试提示词
        
    Returns:
        (是否成功, 响应文本, 延迟时间)
    """
    start_time = time.time()
    
    try:
        # 构造请求
        messages = [{"role": "user", "content": prompt}]
        
        # 执行请求
        response = await litellm.acompletion(
            model=f"{provider}/{model}",
            messages=messages,
            max_tokens=200
        )
        
        # 计算延迟
        latency = time.time() - start_time
        
        # 获取响应文本
        response_text = response.choices[0].message.content
        
        return True, response_text, latency
    except Exception as e:
        logger.error(f"测试 {provider}/{model} 失败: {str(e)}")
        # 计算失败延迟
        latency = time.time() - start_time
        return False, str(e), latency

async def check_llm_service(provider, models=None):
    """测试 LLM 服务的各个方面
    
    Args:
        provider: 服务提供商名称
        models: 要测试的模型列表，如果为None则使用默认模型
    
    Returns:
        (检查结果, 详细信息)
    """
    print(f"\n测试 {provider} 服务:")
    
    # 获取API密钥环境变量名
    api_key_var = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(api_key_var)
    
    if not api_key:
        return False, f"未设置 {api_key_var} 环境变量"
    
    # 如果未指定模型，使用默认模型
    if not models:
        if provider in SUPPORTED_PROVIDERS:
            models = SUPPORTED_PROVIDERS[provider]
        else:
            return False, f"不支持的提供商: {provider}"
    
    # 设置API密钥
    setattr(litellm, f"{provider}_api_key", api_key)
    
    all_tests_passed = True
    test_results = {}
    
    # 对每个模型进行测试
    for model in models:
        print(f"\n  测试模型: {model}")
        model_results = {}
        model_passed = True
        
        # 测试基本功能
        print("  1. 基本功能测试")
        success, response, latency = await test_llm_request(
            provider=provider,
            model=model,
            prompt=TEST_PROMPTS["基本验证"]
        )
        
        if success:
            print(f"    ✅ 基本功能测试通过 (延迟: {latency:.2f}秒)")
            print(f"    响应: {response[:50]}..." if len(response) > 50 else f"    响应: {response}")
            model_results["基本功能"] = {
                "状态": "通过",
                "延迟": f"{latency:.2f}秒",
                "响应": response
            }
        else:
            print(f"    ❌ 基本功能测试失败: {response}")
            model_results["基本功能"] = {
                "状态": "失败",
                "错误": response
            }
            model_passed = False
        
        # 如果基本功能测试通过，继续测试其他能力
        if model_passed:
            # 测试中文能力
            print("  2. 中文处理能力测试")
            success, response, latency = await test_llm_request(
                provider=provider,
                model=model,
                prompt=TEST_PROMPTS["中文能力"]
            )
            
            if success:
                print(f"    ✅ 中文处理测试通过 (延迟: {latency:.2f}秒)")
                print(f"    响应: {response[:50]}..." if len(response) > 50 else f"    响应: {response}")
                model_results["中文能力"] = {
                    "状态": "通过",
                    "延迟": f"{latency:.2f}秒",
                    "响应": response
                }
            else:
                print(f"    ❌ 中文处理测试失败: {response}")
                model_results["中文能力"] = {
                    "状态": "失败",
                    "错误": response
                }
                model_passed = False
            
            # 测试代码能力
            print("  3. 代码生成能力测试")
            success, response, latency = await test_llm_request(
                provider=provider,
                model=model,
                prompt=TEST_PROMPTS["代码能力"]
            )
            
            if success:
                print(f"    ✅ 代码生成测试通过 (延迟: {latency:.2f}秒)")
                print(f"    响应: {response[:50]}..." if len(response) > 50 else f"    响应: {response}")
                model_results["代码能力"] = {
                    "状态": "通过",
                    "延迟": f"{latency:.2f}秒",
                    "响应": response
                }
            else:
                print(f"    ❌ 代码生成测试失败: {response}")
                model_results["代码能力"] = {
                    "状态": "失败",
                    "错误": response
                }
            
            # 测试推理能力
            print("  4. 推理能力测试")
            success, response, latency = await test_llm_request(
                provider=provider,
                model=model,
                prompt=TEST_PROMPTS["推理能力"]
            )
            
            if success:
                print(f"    ✅ 推理能力测试通过 (延迟: {latency:.2f}秒)")
                print(f"    响应: {response[:50]}..." if len(response) > 50 else f"    响应: {response}")
                model_results["推理能力"] = {
                    "状态": "通过",
                    "延迟": f"{latency:.2f}秒",
                    "响应": response
                }
            else:
                print(f"    ❌ 推理能力测试失败: {response}")
                model_results["推理能力"] = {
                    "状态": "失败",
                    "错误": response
                }
        
        # 存储每个模型的测试结果
        test_results[model] = {
            "通过": model_passed,
            "结果详情": model_results
        }
        
        # 如果任何模型测试失败，整体测试标记为失败
        if not model_passed:
            all_tests_passed = False
    
    # 返回测试总结
    if all_tests_passed:
        return True, {
            "测试结果": "所有模型测试通过",
            "详细结果": test_results
        }
    else:
        return False, {
            "测试结果": "部分模型测试失败",
            "详细结果": test_results
        }

async def main():
    """主函数"""
    print("LLM 服务器状态检查")
    print("==================")
    
    # 检查环境变量
    all_set, missing_vars = await check_environment()
    
    # 打印环境变量状态
    print("\n环境变量检查:")
    if all_set:
        print("✅ 至少有一个LLM服务提供商的API密钥已设置")
        available_providers = []
        
        if os.getenv("OPENAI_API_KEY"):
            print("✅ OPENAI_API_KEY: 已设置")
            available_providers.append("openai")
        else:
            print("❌ OPENAI_API_KEY: 未设置")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            print("✅ ANTHROPIC_API_KEY: 已设置")
            available_providers.append("anthropic")
        else:
            print("❌ ANTHROPIC_API_KEY: 未设置")
        
        if os.getenv("DEEPSEEK_API_KEY"):
            print("✅ DEEPSEEK_API_KEY: 已设置")
            available_providers.append("deepseek")
        else:
            print("❌ DEEPSEEK_API_KEY: 未设置")
        
        if os.getenv("GEMINI_API_KEY"):
            print("✅ GEMINI_API_KEY: 已设置")
            available_providers.append("gemini")
        else:
            print("❌ GEMINI_API_KEY: 未设置")
    else:
        print("❌ 没有设置任何LLM服务提供商的API密钥!")
        for var in missing_vars:
            print(f"   - {var}")
        sys.exit(1)
    
    # 初始化结果表格
    final_results = {}
    overall_success = True
    
    # 执行服务检查
    print("\n开始LLM服务检查...")
    
    for provider in available_providers:
        # 每个提供商的健康检查
        success, results = await check_llm_service(provider)
        final_results[provider] = results
        
        if success:
            print(f"\n✅ {provider} 服务检查通过")
        else:
            print(f"\n❌ {provider} 服务检查失败: {results['测试结果']}")
            overall_success = False
    
    # 打印总结
    print("\n====================")
    print("LLM服务检查总结:")
    if overall_success:
        print("✅ 所有可用的LLM服务检查通过")
    else:
        print("❌ 部分LLM服务检查失败")
    
    # 保存详细结果到文件
    with open("llm_service_check_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print("\n详细结果已保存到 llm_service_check_results.json 文件")

if __name__ == "__main__":
    asyncio.run(main()) 