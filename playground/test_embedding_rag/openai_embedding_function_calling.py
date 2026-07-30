"""
对比测试：OpenAI text-embedding-ada-002 vs 智谱 embedding-2

结论：两者调用方式几乎一样，因为智谱 API 兼容 OpenAI 格式。

          OpenAI                          Zhipu
          ──────                          ──────
客户端      openai.OpenAI()                 ZhipuAI()
调用方法     client.embeddings.create(       client.embeddings.create(
              model="xxx",                      model="xxx",
              input=text,                       input=text,
            )                                 )
取向量      response.data[0].embedding       response.data[0].embedding
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---- 1. 加载环境变量 ----
# 从项目根目录加载统一的 .env
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(env_path, override=True)

# ---- 2. OpenAI Embedding 测试 ----
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

texts = [
    "今天天气真好，适合出去散步。",
    "What is Task Decomposition?",
]

print("=" * 60)
print("OpenAI text-embedding-ada-002 测试")
print("=" * 60)

for text in texts:
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    embedding = response.data[0].embedding
    print(f"\n输入: {text}")
    print(f"向量维度: {len(embedding)}")
    print(f"前 5 个值: {embedding[:5]}")

# ---- 3. 与智谱 embedding-2 的对比 ----
print("\n" + "=" * 60)
print("调用方式对比")
print("=" * 60)
print("""
  相同点:
    - 都调用 client.embeddings.create()
    - 参数都有 model 和 input
    - 返回值都是 response.data[0].embedding

  不同点:
    - 模型名:     "text-embedding-ada-002" (OpenAI) vs "embedding-2" (智谱)
    - 向量维度:    1536 (ada-002) vs 1024 (embedding-2)
    - 客户端:      OpenAI(api_key=...) vs ZhipuAI()
    - 收费:        付费 vs 免费额度较多
""")
