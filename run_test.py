import os
from dotenv import load_dotenv
from src.CogletNet import CogletNet
import pytest



if __name__ == "__main__":
    # 调用指定测试函数，便于断点调试
    pytest.main([
        "-s",
        "-v",
        "-W", "ignore::pytest.PytestAssertRewriteWarning",
        "tests/test_cogletnet.py::test_real_think_with_20_coglets"
    ]) 