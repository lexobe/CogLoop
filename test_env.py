import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()
 
# 打印环境变量
print("UPSTASH_URL:", os.getenv("UPSTASH_URL"))
print("UPSTASH_TOKEN:", os.getenv("UPSTASH_TOKEN")) 