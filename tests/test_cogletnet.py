"""
测试 CogletNet 类的功能
"""

import os
import json
import pytest
from unittest.mock import Mock, patch
from src.CogletNet import CogletNet
from dotenv import load_dotenv
from src.log_config import setup_logger

@pytest.fixture
def mock_vector_store():
    """模拟向量存储"""
    mock = Mock()
    mock.recall.return_value = {
        "activated": [
            {
                "id": "cog_001",
                "content": "这是一个测试认元",
                "metadata": {"weight": 0.8}
            },
            {
                "id": "cog_002",
                "content": "这是另一个测试认元",
                "metadata": {"weight": 0.6}
            }
        ]
    }
    return mock

@pytest.fixture
def mock_coglets(mock_vector_store):
    """模拟认元管理"""
    mock = Mock()
    mock.vector_store = mock_vector_store
    mock.recall = mock_vector_store.recall
    mock.create_set.return_value = True
    mock.add.return_value = True
    mock.update_weights.return_value = True
    return mock

@pytest.fixture
def coglet_net(mock_coglets):
    """创建CogletNet实例"""
    with patch('src.CogletNet.VectorStore', return_value=mock_coglets.vector_store), \
         patch('src.CogletNet.Coglets', return_value=mock_coglets):
        net = CogletNet(
            upstash_url="test_url",
            upstash_token="test_token",
            set_id="test_set",
            model="gpt-4.1-nano"  # 使用最便宜的 gpt-4.1-nano 模型
        )
        return net

def test_init(coglet_net):
    """测试初始化"""
    assert coglet_net.set_id == "test_set"
    assert coglet_net.model == "gpt-4.1-nano"
    assert coglet_net.temperature == 0.7
    assert coglet_net.max_retries == 3

def test_format_llm_prompt(coglet_net):
    """测试提示词格式化"""
    query = "测试查询"
    context = [
        {
            "id": "cog_001",
            "content": "测试内容1",
            "metadata": {"weight": 0.8}
        },
        {
            "id": "cog_002",
            "content": "测试内容2",
            "metadata": {"weight": 0.6}
        }
    ]
    
    prompt = coglet_net._format_llm_prompt(query, context)
    
    assert "测试查询" in prompt
    assert "测试内容1" in prompt
    assert "测试内容2" in prompt
    assert "ID=1." in prompt
    assert "ID=2." in prompt
    assert "请基于以上认知进行深入思考" in prompt
    assert "在activated_cog_ids中请使用原始的认元序号" in prompt

@patch('src.CogletNet.completion_with_retries')
def test_call_llm(mock_completion, coglet_net):
    """测试LLM调用"""
    mock_completion.return_value.choices = [
        Mock(message=Mock(content=json.dumps({
            "next_thought": "继续思考",
            "activated_cog_ids": ["cog_001"],
            "log": "测试日志",
            "generated_cog_texts": ["新认元"],
            "function_calls": []
        })))
    ]
    
    response = coglet_net._call_llm("测试提示")
    
    assert isinstance(response, str)
    result = json.loads(response)
    assert "next_thought" in result
    assert "activated_cog_ids" in result
    assert "log" in result
    assert "generated_cog_texts" in result
    assert "function_calls" in result

@patch('src.CogletNet.completion_with_retries')
def test_think(mock_completion, coglet_net):
    """测试思考过程"""
    mock_completion.return_value.choices = [
        Mock(message=Mock(content=json.dumps({
            "next_thought": "继续思考",
            "activated_cog_ids": ["cog_001"],
            "log": "测试日志",
            "generated_cog_texts": ["新认元"],
            "function_calls": []
        })))
    ]
    
    result = coglet_net.think("测试输入", "test_set")
    
    assert isinstance(result, dict)
    assert "next_thought" in result
    assert "activated_cog_ids" in result
    assert "log" in result
    assert "generated_cog_texts" in result
    assert "function_calls" in result

@patch('src.CogletNet.completion_with_retries')
def test_think_loop(mock_completion, coglet_net):
    """测试思考循环"""
    mock_completion.return_value.choices = [
        Mock(message=Mock(content=json.dumps({
            "next_thought": "继续思考",
            "activated_cog_ids": ["cog_001"],
            "log": "测试日志",
            "generated_cog_texts": ["新认元"],
            "function_calls": []
        })))
    ]
    
    results = coglet_net.think_loop("测试输入", "test_set", max_iterations=2)
    
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, dict)
        assert "next_thought" in result
        assert "activated_cog_ids" in result
        assert "log" in result
        assert "generated_cog_texts" in result
        assert "function_calls" in result

def test_call_function(coglet_net):
    """测试函数调用"""
    # 测试X_reply函数
    func_call = {
        "name": "X_reply",
        "args": {
            "post_id": "123",
            "reply_text": "测试回复"
        }
    }
    result = coglet_net.call_function(func_call)
    assert result is True
    
    # 测试未知函数
    func_call = {
        "name": "unknown_function",
        "args": {}
    }
    result = coglet_net.call_function(func_call)
    assert result is None

def test_error_handling(coglet_net):
    """测试错误处理"""
    # 测试JSON解析错误
    with patch('src.CogletNet.completion_with_retries') as mock_completion:
        mock_completion.return_value.choices = [
            Mock(message=Mock(content="invalid json"))
        ]
        result = coglet_net.think("测试输入", "test_set")
        assert result["log"] == "LLM调用失败"
        assert not result["next_thought"]
        assert not result["activated_cog_ids"]
        assert not result["generated_cog_texts"]
        assert not result["function_calls"]
    
    # 测试LLM调用失败
    with patch('src.CogletNet.completion_with_retries') as mock_completion:
        mock_completion.side_effect = Exception("API调用失败")
        result = coglet_net.think("测试输入", "test_set")
        assert result["log"] == "LLM调用失败"
        assert not result["next_thought"]
        assert not result["activated_cog_ids"]
        assert not result["generated_cog_texts"]
        assert not result["function_calls"]

def test_real_think_with_20_coglets():
    load_dotenv()
    set_id = "test_coglets"
    
    # 获取日志记录器
    logger = setup_logger("CogletNet")
    logger.debug("开始测试 test_real_think_with_20_coglets")

    # 初始化真实 CogletNet
    net = CogletNet(
        upstash_url=os.getenv("UPSTASH_URL"),
        upstash_token=os.getenv("UPSTASH_TOKEN"),
        set_id=set_id,
        model="gpt-4o-mini",  # 使用 litellm 官方支持的模型
        top_k=10
    )
    logger.debug("CogletNet 初始化完成")

    net.coglets.create_set(set_id)
    logger.debug(f"创建认元集合: {set_id}")

    # 20条认元
    coglet_texts = [
        "人工智能通过深度学习和自然语言处理技术，能够自动分析海量数据并提取关键信息，为人类决策提供数据支持。它不仅提高了工作效率，还能发现人类可能忽略的潜在模式和关联，但同时也带来了数据隐私和算法偏见等伦理挑战。",
        "团队协作的核心在于建立清晰的沟通机制和明确的责任分工。通过定期的同步会议、透明的任务分配和有效的反馈机制，团队成员能够更好地理解彼此的工作进展，减少信息不对称，提高整体协作效率。",
        "持续学习是个人在快速变化的时代保持竞争力的关键。通过系统性地学习新知识、技能和方法，个人能够不断适应新的工作环境和挑战，同时培养批判性思维和创新能力，为职业发展打下坚实基础。",
        "健康的生活方式包括规律的作息、均衡的饮食和适度的运动。这些习惯不仅能够提高身体素质，还能改善心理状态，增强工作专注力和创造力。研究表明，保持健康的生活方式可以显著提升工作效率和生活质量。",
        "有效的目标设定需要遵循SMART原则：具体、可衡量、可实现、相关性和时限性。通过设定清晰的目标，个人和团队能够更好地规划行动步骤，跟踪进展，及时调整策略，最终实现预期成果。",
        "积极的反馈机制是团队建设的重要组成部分。通过及时、具体、建设性的反馈，团队成员能够了解自己的表现，明确改进方向，增强工作动力。同时，良好的反馈文化也有助于建立开放、信任的团队氛围。",
        "技术创新正在深刻改变我们的生活方式和工作方式。从人工智能到区块链，从物联网到量子计算，这些技术不仅提高了效率，也带来了新的商业模式和社会形态。然而，技术发展也带来了隐私保护、数字鸿沟等社会问题。",
        "时间管理的本质是优先级管理。通过识别重要且紧急的任务，合理分配时间和精力，使用番茄工作法等工具，个人能够提高工作效率，减少压力，在有限的时间内完成更多有价值的工作。",
        "情绪管理能力是职场成功的重要因素。通过识别自己的情绪状态，理解情绪产生的原因，采用适当的调节策略，个人能够更好地处理工作压力，建立良好的人际关系，做出理性的决策。",
        "跨学科知识整合是解决复杂问题的关键。通过结合不同领域的理论、方法和工具，个人能够从多角度分析问题，发现创新解决方案。这种思维方式特别适合处理涉及技术、社会、经济等多个维度的复杂问题。",
        "教育的本质是激发学习者的内在潜能。通过创设良好的学习环境，提供个性化的学习支持，鼓励探索和实验，教育者能够帮助学习者发现自己的兴趣和优势，培养终身学习的能力和习惯。",
        "开放心态是适应变化的基础。通过保持对新事物的好奇心，愿意接受不同的观点和想法，个人能够更好地适应环境变化，发现新的机会，在变革中找到自己的位置和发展方向。",
        "数据安全和隐私保护是数字时代的重要议题。随着数据成为重要资产，如何平衡数据利用和隐私保护，建立安全的数据管理体系，防范数据泄露和滥用，成为个人、组织和社会共同面临的挑战。",
        "有效的会议管理需要明确的议程、时间控制和行动计划。通过会前准备、会中引导和会后跟进，会议组织者能够确保会议高效进行，达成预期目标，避免时间浪费和无效讨论。",
        "阅读经典著作能够提升思维深度和广度。通过系统性地阅读不同领域的经典作品，个人能够了解人类思想的发展历程，培养批判性思维，建立知识体系，提升表达和写作能力。",
        "适度运动对身心健康有显著益处。通过规律的有氧运动和力量训练，个人能够提高身体素质，增强免疫力，改善睡眠质量，缓解压力，提升工作效率和创造力。",
        "领导力不仅仅是管理，更是激发团队潜能。通过建立清晰的愿景，提供必要的支持，培养团队成员的能力，优秀的领导者能够激发团队的创造力和执行力，实现组织目标。",
        "失败是创新过程中不可或缺的一部分。通过正确看待失败，从中总结经验教训，调整策略和方法，个人和团队能够不断改进，最终实现突破。失败不是终点，而是成长的机会。",
        "善用工具和技术能够显著提升工作效率。通过选择合适的软件工具，掌握快捷键和自动化技巧，建立个人知识管理系统，个人能够减少重复性工作，提高工作质量和效率。",
        "同理心是建立良好人际关系的基础。通过理解他人的感受和需求，站在对方的角度思考问题，个人能够建立信任，化解冲突，促进有效沟通，创造和谐的工作和生活环境。"
    ]
    for text in coglet_texts:
        net.coglets.add(set_id, text)
    logger.debug(f"添加了 {len(coglet_texts)} 条认元")

    # 进行一次思考
    input_text = "在人工智能快速发展的时代，如何帮助团队成员保持竞争力并建立高效的工作方式？"
    logger.debug(f"开始思考，输入: {input_text}")
    result = net.think(input_text, set_id)
    logger.debug("思考完成")

    # 打印结果
    print("\n思考结果：")
    print(f"下一轮思考：{result['next_thought']}")
    print(f"\n激活的认元：{result['activated_cog_ids']}")
    print(f"\n思考日志：{result['log']}")
    print(f"\n生成的认元：{result['generated_cog_texts']}")
    print(f"\n函数调用：{result['function_calls']}")

    # 断言（可选）
    assert isinstance(result, dict)
    assert "next_thought" in result
    assert "activated_cog_ids" in result
    assert len(result["activated_cog_ids"]) > 0
    
    logger.debug("测试完成")

if __name__ == "__main__":
    pytest.main(["-v", "test_cogletnet.py"]) 