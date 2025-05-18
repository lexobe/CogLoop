"""
LLM 服务器状态检查工具

功能：
1. 检查 API 密钥配置
2. 测试服务器连接
3. 测试响应时间
4. 测试基本功能
5. 生成状态报告
"""

import os
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
from litellm import completion_with_retries
from litellm import ModelResponse
from src.log_config import setup_logger

logger = setup_logger("check_llm")

class LLMChecker:
    """LLM 服务器状态检查器"""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        timeout: int = 30,
        test_prompts: Optional[list] = None
    ):
        """
        初始化检查器
        
        Args:
            model: LLM模型名称
            max_retries: 最大重试次数
            timeout: 超时时间（秒）
            test_prompts: 测试提示词列表
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.test_prompts = test_prompts or [
            "你好，请回复一个简单的问候。",
            "1+1=?",
            "用一句话描述人工智能。"
        ]
        
    def check_api_key(self) -> Dict[str, Any]:
        """
        检查 API 密钥配置
        
        Returns:
            Dict[str, Any]: 检查结果
        """
        api_key = os.getenv("OPENAI_API_KEY")
        return {
            "status": "ok" if api_key else "error",
            "message": "API密钥已配置" if api_key else "未找到API密钥",
            "api_key_length": len(api_key) if api_key else 0
        }
        
    def test_connection(self) -> Dict[str, Any]:
        """
        测试服务器连接
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        start_time = time.time()
        try:
            response = completion_with_retries(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                max_retries=1,
                timeout=self.timeout
            )
            latency = time.time() - start_time
            
            return {
                "status": "ok",
                "message": "连接成功",
                "latency": round(latency, 3),
                "response": response.choices[0].message.content
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"连接失败: {str(e)}",
                "latency": round(time.time() - start_time, 3)
            }
            
    def test_response_time(self, num_requests: int = 3) -> Dict[str, Any]:
        """
        测试响应时间
        
        Args:
            num_requests: 测试请求数量
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        latencies = []
        errors = []
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                response = completion_with_retries(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                    max_retries=1,
                    timeout=self.timeout
                )
                latency = time.time() - start_time
                latencies.append(latency)
            except Exception as e:
                errors.append(str(e))
                
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
        else:
            avg_latency = max_latency = min_latency = 0
            
        return {
            "status": "ok" if not errors else "error",
            "message": f"完成 {num_requests} 次测试",
            "avg_latency": round(avg_latency, 3),
            "max_latency": round(max_latency, 3),
            "min_latency": round(min_latency, 3),
            "errors": errors
        }
        
    def test_functionality(self) -> Dict[str, Any]:
        """
        测试基本功能
        
        Returns:
            Dict[str, Any]: 测试结果
        """
        results = []
        
        for prompt in self.test_prompts:
            try:
                response = completion_with_retries(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_retries=1,
                    timeout=self.timeout
                )
                results.append({
                    "prompt": prompt,
                    "status": "ok",
                    "response": response.choices[0].message.content
                })
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "status": "error",
                    "error": str(e)
                })
                
        return {
            "status": "ok" if all(r["status"] == "ok" for r in results) else "error",
            "message": f"完成 {len(self.test_prompts)} 个功能测试",
            "results": results
        }
        
    def run_all_checks(self) -> Dict[str, Any]:
        """
        运行所有检查
        
        Returns:
            Dict[str, Any]: 完整检查报告
        """
        start_time = time.time()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "checks": {
                "api_key": self.check_api_key(),
                "connection": self.test_connection(),
                "response_time": self.test_response_time(),
                "functionality": self.test_functionality()
            }
        }
        
        # 计算总体状态
        all_checks = report["checks"].values()
        report["overall_status"] = "ok" if all(check["status"] == "ok" for check in all_checks) else "error"
        report["total_time"] = round(time.time() - start_time, 3)
        
        return report
        
    def print_report(self, report: Optional[Dict[str, Any]] = None) -> None:
        """
        打印检查报告
        
        Args:
            report: 检查报告，如果为None则运行新的检查
        """
        if report is None:
            report = self.run_all_checks()
            
        logger.info("\n=== LLM Server Check Report ===")
        logger.info(f"Time: {report['timestamp']}")
        logger.info(f"Model: {report['model']}")
        logger.info(f"Status: {report['overall_status']}")
        logger.info(f"Total time: {report['total_time']}s")
        logger.info("\nDetails:")
        
        for check_name, check_result in report["checks"].items():
            logger.info(f"\n{check_name}:")
            logger.info(f"  Status: {check_result['status']}")
            logger.info(f"  Message: {check_result['message']}")
            
            if check_name == "response_time":
                logger.info(f"  Avg latency: {check_result['avg_latency']}s")
                logger.info(f"  Max latency: {check_result['max_latency']}s")
                logger.info(f"  Min latency: {check_result['min_latency']}s")
                
            if check_name == "functionality":
                logger.info("  Function test results:")
                for result in check_result["results"]:
                    logger.info(f"    - Prompt: {result['prompt']}")
                    logger.info(f"      Status: {result['status']}")
                    if result["status"] == "ok":
                        logger.info(f"      Response: {result['response']}")
                    else:
                        logger.info(f"      Error: {result['error']}")

if __name__ == "__main__":
    # 创建检查器
    checker = LLMChecker(
        model="gpt-3.5-turbo",
        max_retries=3,
        timeout=30
    )
    
    # 运行检查并打印报告
    checker.print_report() 