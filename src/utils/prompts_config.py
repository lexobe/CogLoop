"""
集中管理所有的prompt和参数配置
"""
from typing import Dict, Any, List, Optional

# ==============================================================================
# LLM配置参数
# ==============================================================================
LLM_CONFIG = {
    # 默认LLM提供商
    "DEFAULT_PROVIDER": "openai",
    
    # 各提供商的默认模型
    "DEFAULT_MODELS": {
        "openai": "gpt-4-turbo-preview",
        "anthropic": "claude-3-opus-20240229",
        "deepseek": "deepseek-chat",
        "gemini": "gemini-pro"
    },
    
    # 请求参数
    "DEFAULT_PARAMETERS": {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 2000
    }
}

# ==============================================================================
# 认知循环参数
# ==============================================================================
COGNITIVE_LOOP_CONFIG = {
    # 权重更新参数
    "beta": 0.8,  # 旧记忆残留因子
    "gamma": 1.0,  # 新调用强化增益系数
    "b": 0.1,     # 时间敏感系数
    
    # 搜索参数
    "default_top_k": 5,
    "default_min_score": 0.7
}

# ==============================================================================
# 向量存储参数
# ==============================================================================
VECTOR_STORE_CONFIG = {
    # 默认嵌入模型
    "DEFAULT_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    
    # 黄金分割比例（用于结果过滤）
    "GOLDEN_RATIO": 0.618,
    
    # 默认搜索参数
    "DEFAULT_TOP_K": 5,
    "DEFAULT_MIN_SCORE": 0.7,
    
    # 健康检查测试文本
    "HEALTH_CHECK_TEXT": "健康检查测试"
}

# ==============================================================================
# 系统提示词模板
# ==============================================================================
SYSTEM_PROMPTS = {
    "DEFAULT": """你是一个智能助手，基于认知循环架构，可以理解用户的需求并提供有用的回答。
你可以访问相关记忆来帮助你生成更好的回答。
请根据当前输入和提供的相关记忆来回答问题。""",
    
    "CREATIVE": """你是一个富有创造力的AI助手，基于认知循环架构，擅长生成创意内容。
利用提供的相关记忆，你可以创作出连贯且有创意的内容。
请根据当前输入和相关记忆，发挥你的创造力。""",
    
    "ACADEMIC": """你是一个学术研究助手，基于认知循环架构，专注于提供准确和有深度的学术信息。
利用提供的相关记忆，你可以分析学术问题并提供深入的见解。
请根据当前输入和相关记忆，给出学术性的回答，必要时引用来源。"""
}

# ==============================================================================
# 上下文构建模板
# ==============================================================================
CONTEXT_TEMPLATES = {
    "DEFAULT": """## 相关记忆
{memories}

## 当前输入
{input_text}

请根据以上信息提供回答。""",
    
    "MEMORY_ITEM": """记忆 {index}（权重: {weight:.2f}, 相似度: {score:.2f}）: {content}"""
}

# ==============================================================================
# 测试用提示词
# ==============================================================================
TEST_PROMPTS = {
    "基本验证": "请简单介绍一下自己",
    "中文能力": "请用中文解释什么是认知循环架构，并举例说明其应用",
    "代码能力": "请写一个Python函数，实现斐波那契数列的计算",
    "推理能力": "如果一个球从10米高的地方落下，每次弹起的高度是原来的一半，那么它总共会经过多少距离？",
    "多轮对话": "我们来玩个角色扮演游戏，你扮演一个历史学家，我想向你请教有关中国历史的问题"
}

def get_default_llm_model(provider: str) -> str:
    """获取指定提供商的默认模型"""
    return LLM_CONFIG["DEFAULT_MODELS"].get(provider, LLM_CONFIG["DEFAULT_MODELS"]["openai"])

def get_system_prompt(prompt_type: str = "DEFAULT") -> str:
    """获取系统提示词"""
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["DEFAULT"])

def build_context(memories: List[Dict[str, Any]], input_text: str, template: str = "DEFAULT") -> str:
    """构建上下文
    
    Args:
        memories: 记忆列表，每个元素包含content, weight, score等信息
        input_text: 当前输入文本
        template: 模板类型
        
    Returns:
        构建好的上下文字符串
    """
    context_template = CONTEXT_TEMPLATES.get(template, CONTEXT_TEMPLATES["DEFAULT"])
    memory_template = CONTEXT_TEMPLATES["MEMORY_ITEM"]
    
    memory_texts = []
    for i, memory in enumerate(memories):
        memory_text = memory_template.format(
            index=i+1,
            weight=memory.get("weight", 0),
            score=memory.get("score", 0),
            content=memory.get("content", "")
        )
        memory_texts.append(memory_text)
    
    memories_text = "\n".join(memory_texts) if memory_texts else "无相关记忆"
    
    return context_template.format(
        memories=memories_text,
        input_text=input_text
    ) 