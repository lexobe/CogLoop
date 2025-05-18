"""
CogletNet：认知网络核心实现

实现认知循环（Think Loop）的核心功能，包括：
1. 感知（Perception）：接收和处理输入信息
2. 思考（Thinking）：基于已有认知进行推理和决策
3. 记忆（Memory）：存储和更新认知
4. 行动（Action）：执行决策并产生输出
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import json
import uuid
import os
from .vector_store import VectorStore
from .coglets import Coglets
from litellm import completion_with_retries
from .mam import MAM
from .id_generator import vector_id
from src.log_config import setup_logger

logger = setup_logger("CogletNet")

class CogletNet:
    """认知网络核心类，实现认知循环"""
    
    def __init__(
        self,
        upstash_url: str,
        upstash_token: str,
        set_id: Optional[str] = None,
        model: str = "4o-mini",  # 默认使用 4o-mini
        max_retries: int = 3,
        temperature: float = 0.7,
        beta: float = 0.85,      # 时间衰减系数
        gamma: float = 0.3,      # 访问增强系数
        b: float = 0.05,         # 基础衰减率
        initial_weight: float = 0.5,  # 初始权重
        golden_ratio: float = 0.618,  # 黄金分割比例
        top_k: int = 10,  # 新增top_k参数
        max_think_cycles: int = 5,    # 最大思考循环次数
        think_threshold: float = 0.7,  # 思考阈值，低于此值认为思考完成
        role_prompt: Optional[str] = None,  # 角色定义
        format_prompt: Optional[str] = None,  # 输出格式定义
        thinking_prompt: Optional[str] = None,  # 思考指导
        example_prompt: Optional[str] = None,  # 示例说明
    ):
        """
        初始化认知网络
        
        Args:
            upstash_url: Upstash 服务地址
            upstash_token: Upstash 访问令牌
            set_id: 认知集合ID，如果为None则自动生成
            model: LLM模型名称，默认使用4o-mini
            max_retries: LLM调用最大重试次数
            temperature: LLM温度参数
            beta: 时间衰减系数
            gamma: 访问增强系数
            b: 基础衰减率
            initial_weight: 初始权重
            golden_ratio: 黄金分割比例
            max_think_cycles: 最大思考循环次数
            think_threshold: 思考阈值，低于此值认为思考完成
            role_prompt: 角色定义提示词
            format_prompt: 输出格式定义提示词
            thinking_prompt: 思考指导提示词
            example_prompt: 示例说明提示词
        """
        self.vector_store = VectorStore(upstash_url, upstash_token)
        self.coglets = Coglets(
            self.vector_store,
            beta=beta,
            gamma=gamma,
            b=b,
            initial_weight=initial_weight,
            golden_ratio=golden_ratio,
            top_k=top_k
        )
        
        # 设置LLM参数
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        
        self.max_think_cycles = max_think_cycles
        self.think_threshold = think_threshold
        
        # 设置认知集合ID
        self.set_id = set_id or f"coglet_{uuid.uuid4().hex[:8]}"
        self.coglets.create_set(self.set_id, f"认知集合 {self.set_id}")
        
        # 设置LLM提示词
        self.role_prompt = role_prompt or "你是一个专业的认知分析系统，擅长从输入内容中提取关键信息并生成结构化的思考。你必须严格按照指定的JSON格式返回结果。"
        
        self.format_prompt = format_prompt or (
            "你必须严格按照以下JSON格式返回结果：\n"
            "{\n"
            '  "next_thought": "下一轮思考内容",\n'
            '  "activated_cog_ids": ["实际使用到的认元序号"],\n'
            '  "log": "思考日志",\n'
            '  "generated_cog_texts": ["生成的认元文本列表"],\n'
            '  "function_calls": [{"name": "函数名", "args": {"参数名": "参数值"}}]\n'
            "}"
        )
        
        self.thinking_prompt = thinking_prompt or (
            "在思考过程中，请注意以下几点：\n"
            "1. 认元应当是基于输入内容的抽象性观点与模式总结\n"
            "2. 生成的认元文本应当简洁、清晰、具有普遍性\n"
            "3. 思考日志应当记录重要的推理过程和决策依据\n"
            "4. 函数调用应当明确具体，包含必要的参数\n"
            "5. 在activated_cog_ids中列出实际用于生成回答的认元ID\n"
            "6. 必须严格按照指定的JSON格式返回结果"
        )
        
        self.example_prompt = example_prompt or (
            """示例输入：'人工智能正在改变我们的生活方式'
示例输出：
{
  "next_thought": "这种改变主要体现在哪些具体方面？",
  "activated_cog_ids": [1, 2],
  "log": "识别到技术变革主题，需要进一步探讨具体影响",
  "generated_cog_texts": ["技术变革往往从日常生活开始", "AI的影响具有普遍性和深远性"],
  "function_calls": []
}
"""
        )
        
    def _format_llm_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """格式化提示词"""
        prompt = f"Based on the following cognitions, think and answer: {query}\n\nRelated cognitions:\n"
        
        for i, cog in enumerate(context, 1):
            prompt += f"ID={i}. {cog['content']}\n"
        
        prompt += "\nPlease think deeply based on the above cognitions and give your insights. Use the original cognitive element sequence number in activated_cog_ids."
        return prompt
        
    def _get_system_prompt(self) -> str:
        """
        获取完整的系统提示词
        
        Returns:
            str: 完整的系统提示词
        """
        return f"{self.role_prompt}\n\n{self.format_prompt}\n\n{self.thinking_prompt}\n\n{self.example_prompt}"
        
    def think(self, input_text: str, set_id: str) -> Dict[str, Any]:
        """
        单次思考过程
        
        Args:
            input_text: 输入文本
            set_id: 认元集合ID
            
        Returns:
            Dict[str, Any]: 思考结果
        """
        logger.info(f"Start think: input={input_text}, set={set_id}")
        # 1. 认元激活
        results = self.coglets.recall(set_id, input_text)
        logger.debug(f"Recall result: {results}")
        activated = results["activated"]
        logger.info(f"Activated count={len(activated)}")
        logger.debug(f"Activated content={[c['content'] for c in activated]}")
        
        # 2. 构造上下文
        prompt = self._format_llm_prompt(input_text, activated)
        logger.debug(f"Prompt: {prompt}")
        
        # 3. 调用 LLM 生成响应
        try:
            response = self._call_llm(prompt)
            logger.info("LLM call success")

        except Exception as e:
            logger.error(f"LLM call error: {e}")
            response = '{"next_thought": "", "activated_cog_ids": [], "log": "LLM call failed", "generated_cog_texts": [], "function_calls": []}'
        
        # 4. 解析响应
        try:
            result = json.loads(response)
            logger.debug(f"Parsed response: {result}")
            if not result or not isinstance(result, dict) or result == {}:
                raise ValueError('Empty response')
            
            # 将数组序号转换为认元ID
            activated_ids = []
            for idx in result.get("activated_cog_ids", []):
                try:
                    idx = int(str(idx).replace("ID=", ""))
                    if 1 <= idx <= len(activated):
                        activated_ids.append(activated[idx-1]["id"])
                except (ValueError, IndexError):
                    continue
            
            result = {
                "next_thought": result.get("next_thought", ""),
                "activated_cog_ids": activated_ids,
                "log": result.get("log", ""),
                "generated_cog_texts": result.get("generated_cog_texts", []),
                "function_calls": result.get("function_calls", [])
            }
        except Exception as e:
            logger.warning(f"Parse response failed: {e}")
            result = {
                "next_thought": "",
                "activated_cog_ids": [],
                "log": "Parse response failed",
                "generated_cog_texts": [],
                "function_calls": []
            }
        
        # 5. 更新认元权重
        if result.get("activated_cog_ids"):
            logger.info(f"Update weight ids: {result['activated_cog_ids']}")
            self.coglets.update_weights(result["activated_cog_ids"])
        
        # 6. 写入新认元
        for cog_text in result.get("generated_cog_texts", []):
            logger.info(f"Add coglet: {cog_text}")
            self.coglets.add(set_id, cog_text, {"source": "generated"})
        
        # 7. 执行函数调用
        if result.get("function_calls"):
            for func_call in result["function_calls"]:
                logger.info(f"Call function: {func_call}")
                self.call_function(func_call)
        
        return result

    def think_loop(self, input_text: str, set_id: str, max_iterations: int = 5) -> List[Dict[str, Any]]:
        """
        循环思考过程
        
        Args:
            input_text: 初始输入文本
            set_id: 认元集合ID
            max_iterations: 最大迭代次数
            
        Returns:
            List[Dict[str, Any]]: 思考过程记录
        """
        results = []
        current_input = input_text
        
        for _ in range(max_iterations):
            result = self.think(current_input, set_id)
            results.append(result)
            
            if not result.get("next_thought"):
                break
                
            current_input = result["next_thought"]
        
        return results

    def call_function(self, func_call: Dict[str, Any]) -> Any:
        """
        执行函数调用
        
        Args:
            func_call: 函数调用信息
            
        Returns:
            Any: 函数执行结果
        """
        func_name = func_call.get("name")
        args = func_call.get("args", {})
        
        if func_name == "X_reply":
            return self._x_reply(args)
        else:
            logger.warning(f"Unknown function: {func_name}")
            return None

    def _call_llm(self, prompt: str) -> str:
        """
        调用 LLM 生成响应
        
        Args:
            prompt: 输入提示
            
        Returns:
            str: LLM 响应
        """
        # 这里使用 litellm 调用 LLM
        try:
            response = completion_with_retries(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_retries=self.max_retries,
                temperature=self.temperature,
                response_format={"type": "json_object"}  # 确保返回JSON格式
            )
            content = response.choices[0].message.content
            # 确保返回的是有效的JSON字符串
            json.loads(content)  # 验证JSON格式
            return content
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return '{"next_thought": "", "activated_cog_ids": [], "log": "LLM call failed", "generated_cog_texts": [], "function_calls": []}'

    def _x_reply(self, args: Dict[str, Any]) -> bool:
        """
        执行 X 回复函数
        
        Args:
            args: 函数参数
            
        Returns:
            bool: 是否执行成功
        """
        post_id = args.get("post_id")
        reply_text = args.get("reply_text")
        
        if not post_id or not reply_text:
            return False
            
        # 这里实现实际的 X 回复逻辑
        logger.info(f"Reply post {post_id}: {reply_text}")
        return True

if __name__ == "__main__":
    # 使用示例
    import os
    
    # 自定义提示词
    custom_role_prompt = "你是一个专注于技术分析的认知系统，擅长从技术角度分析问题。你必须严格按照指定的JSON格式返回结果。"
    custom_format_prompt = (
        "你必须严格按照以下JSON格式返回结果：\n"
        "{\n"
        '  "next_thought": "下一轮思考内容",\n'
        '  "activated_cog_ids": ["实际使用到的认元ID列表"],\n'
        '  "log": "思考日志",\n'
        '  "generated_cog_texts": ["生成的认元文本列表"],\n'
        '  "function_calls": [{"name": "函数名", "args": {"参数名": "参数值"}}]\n'
        "}"
    )
    custom_thinking_prompt = (
        "在思考过程中，请注意以下几点：\n"
        "1. 关注技术实现细节和架构设计\n"
        "2. 分析技术选型的优缺点\n"
        "3. 考虑性能和可扩展性\n"
        "4. 记录技术决策依据\n"
        "5. 在activated_cog_ids中列出实际用于生成回答的认元ID\n"
        "6. 必须严格按照指定的JSON格式返回结果"
    )
    custom_example_prompt = (
        """示例输入：'微服务架构的优势是什么？'
示例输出：
{
  "next_thought": "这些优势在什么场景下最明显？",
  "activated_cog_ids": ["tech_001", "tech_002"],
  "log": "识别到架构设计主题，需要进一步探讨应用场景",
  "generated_cog_texts": ["微服务架构支持独立部署和扩展", "服务解耦提高了系统灵活性"],
  "function_calls": []
}
"""
    )
    
    # 初始化认知网络
    net = CogletNet(
        upstash_url=os.getenv("UPSTASH_URL"),
        upstash_token=os.getenv("UPSTASH_TOKEN"),
        set_id="test_set",  # 可选，不提供则自动生成
        model="4o-mini",  # 使用 4o-mini 模型
        role_prompt=custom_role_prompt,
        format_prompt=custom_format_prompt,
        thinking_prompt=custom_thinking_prompt,
        example_prompt=custom_example_prompt
    )
    
    # 执行认知循环
    result = net.think_loop("What is cognitive network?", "test_set")
    
    logger.info(json.dumps(result, ensure_ascii=False, indent=2)) 