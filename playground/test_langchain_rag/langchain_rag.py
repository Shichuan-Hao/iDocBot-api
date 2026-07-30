#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 LangChain 中使用 GLM-4 实现基于 Chroma 向量数据库的 RAG 完整过程。

RAG 流程（参考流程图）:
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Indexing（索引）:                                                  │
│    Documents → Chunking → Document chunks → Vectorize & Store        │
│    → Vector Database                                                │
│                                                                     │
│ 2. Retrieval & Generation（检索与生成）:                             │
│    User Query → Vectorize & Search → Retrieve Relevant Contexts      │
│    → Augment Prompt → LLM Generate → Response                       │
└─────────────────────────────────────────────────────────────────────┘

============================================================
核心概念速览：
- Embedding（嵌入/向量化）：把一段文字转换成一串数字（向量），语义相近的文本向量也相近
- Vector Database（向量数据库）：用"最近邻搜索"快速找到与查询向量最相似的文档
- RAG（Retrieval-Augmented Generation）：先检索相关知识，再让 LLM 基于这些知识回答，减少幻觉
- LCEL（LangChain Expression Language）：用 | 管道符号串联处理步骤，构建链式调用
============================================================
"""

import os
from pathlib import Path

import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client
from zhipuai import ZhipuAI

# ----- 环境准备 -----
# 在控制台先安装依赖：
# pip install --upgrade langchain langchain-community langchainhub \
#     httpx httpx-sse PyJWT langchain-chroma bs4 python-dotenv

# 从项目根目录下的 .env 文件中读取 API Key 等环境变量
# .env 文件内容示例：
#   ZHIPUAI_API_KEY=你的智谱API密钥
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(env_path, override=True)


# ========================= 辅助工具 =========================

class EmbeddingGenerator:
    """
    将文本转换为向量（Embedding），对接智谱 GLM Embedding-2 模型。

    什么是 Embedding？
      把一段文字映射为一串 1024 维的浮点数。语义相近的句子，向量在空间中距离也近；
      语义无关的句子，向量距离远。这是"语义搜索"的基础。

    为什么需要 embed_documents 和 embed_query 两个方法？
      LangChain 规定嵌入模型必须实现这两个方法：
      - embed_documents: 批量处理文档块（用于建索引）
      - embed_query:     处理单条用户查询（用于检索）
      实现后可直接传给 Chroma 的 embedding_function 参数。
    """

    def __init__(self, model_name="embedding-2"):
        """
        参数:
          model_name: 智谱的嵌入模型名称，可选 "embedding-2" / "embedding-3"
                      "embedding-2" 输出 1024 维向量，免费额度较多，适合入门
        """
        self.model_name = model_name
        # ZhipuAI 客户端，自动从环境变量 ZHIPUAI_API_KEY 读取密钥
        self.client = ZhipuAI()

    def embed_documents(self, texts):
        """
        批量将文档块文本转为向量列表。

        参数:
          texts: List[str]  要向量化的文档文本列表，通常来自 Chunking 后的片段
        返回:
          List[List[float]] 每段文本对应一个 1024 维的浮点向量
        """
        embeddings = []
        for text in texts:
            # 调用智谱 Embedding API，单次请求处理一条文本
            response = self.client.embeddings.create(
                model=self.model_name,  # 嵌入模型名称
                input=text,             # 要嵌入的文本内容
            )
            if hasattr(response, 'data') and response.data:
                # response.data[0].embedding 是一个 1024 维的 float 列表
                embeddings.append(response.data[0].embedding)
            else:
                raise RuntimeError(f"获取文档嵌入向量失败，文本: {text[:50]}...")
        return embeddings

    def embed_query(self, query):
        """
        将用户查询转为向量（用于检索时计算相似度）。

        参数:
          query: str  用户输入的问题
        返回:
          List[float] 问题对应的 1024 维浮点向量
        """
        response = self.client.embeddings.create(model=self.model_name, input=query)
        if hasattr(response, 'data') and response.data:
            return response.data[0].embedding
        raise RuntimeError(f"获取查询嵌入向量失败，查询: {query}")


def format_docs(docs):
    """
    把检索到的文档对象列表拼接成一个纯文本字符串。

    参数:
      docs: List[Document]  检索器返回的相关文档对象列表
            每个 Document 对象有两个属性：
              - page_content: str      文档的文本内容
              - metadata: dict         文档的元数据（如来源 URL、页码等）
    返回:
      str  用两个换行符 "\n\n" 分隔拼接后的上下文字符串
    为什么用 "\n\n" 分隔？
      让 LLM 能清晰区分不同文档片段，避免内容混淆。
    """
    return "\n\n".join(doc.page_content for doc in docs)


# ========================= 1. Indexing（索引阶段） =========================
# 目标：把外部知识（网页文档）预处理后存入向量数据库，供后续检索使用。

# ----- Step 1. Load Documents: 加载原始文档 -----
loader = WebBaseLoader(
    # web_paths: 要抓取的网页 URL 列表（元组形式也可）
    # 这里使用 Lilian Weng 关于 AI Agent 的博客文章作为知识库
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),

    # bs_kwargs: 传递给 BeautifulSoup（HTML 解析器）的额外参数
    bs_kwargs=dict(
        # parse_only: SoupStrainer 的解析范围限制
        #   只提取 CSS class 为 "post-content"、"post-title"、"post-header" 的内容
        #   好处：过滤掉导航栏、广告、侧边栏等无关内容，提高检索质量
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()   # 返回 List[Document]，每个 Document 包含整篇文章

# ----- Step 2. Chunking: 文档切分 -----
# 为什么需要 Chunking？
#   1) LLM 的上下文窗口有限，不能一次塞入整篇文章
#   2) 小块可以更精确地检索到与问题相关的部分
#   3) 向量化时，太大的文本会丢失细节语义
text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size: 每个文本块的最大字符数（1000 个字符 ≈ 300~500 个中文字）
    #   太小：信息不完整，检索容易漏掉上下文
    #   太大：检索不够精准，且超出 LLM 上下文限制
    chunk_size=1000,

    # chunk_overlap: 相邻两个块之间重叠的字符数
    #   为什么需要重叠？
    #     防止一句话被切断在两个块的边界上，导致语义断裂。
    #     比如 "苹果公司的总部位于" 被切在块1末尾，"加州库比蒂诺" 被切在块2开头，
    #     有重叠就能保证关键的相邻信息被任一 chunk 完整捕获。
    #   通常设为 chunk_size 的 10% ~ 20%
    chunk_overlap=200,

    # 其他常用参数（本例使用默认值）：
    #   separators:   分隔字符的优先级列表，默认 ["\n\n", "\n", " ", ""]
    #                 按优先级尝试在这些位置切分，尽量保持语义完整
    #   length_function: 计算文本长度的函数，默认 len（字符数）
    #   is_separator_regex: 分隔符是否为正则，默认 False
)
chunks = text_splitter.split_documents(documents)  # 返回 List[Document]，每个 Document 是一个 chunk

# 可取消注释，查看切分效果
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}: {chunk.page_content[:80]}...")

# ----- Step 3. Embedding: 将文档块转为向量 -----
# 创建嵌入生成器，指定模型为智谱的 embedding-2
embedding_generator = EmbeddingGenerator(model_name="embedding-2")
# 提取每个 chunk 的纯文本内容
chunk_texts = [chunk.page_content for chunk in chunks]

# ----- Step 4. Vectorize & Store: 存入向量数据库 -----
chroma_store = Chroma(
    # collection_name: 集合名，相当于传统数据库中的"表"
    #   同一个 Chroma 实例可以有多个 collection，用于区分不同项目/知识库
    collection_name="example_collection",

    # embedding_function: 嵌入函数对象
    #   当写入文本或查询时，Chroma 会自动调用该对象的 embed_documents / embed_query
    #   注意：传入的是实例，不是调用结果。必须实现 embed_documents 和 embed_query 方法
    embedding_function=embedding_generator,

    # create_collection_if_not_exists: 如果集合不存在则自动创建
    #   True:  首次运行自动建表，适合开发和学习
    #   False: 集合必须预先存在，否则报错，适合生产环境
    create_collection_if_not_exists=True,

    # 其他常用参数（本例使用默认值）：
    #   persist_directory: 持久化目录路径，不设置则数据仅存于内存，程序结束后丢失
    #                      如 persist_directory="./chroma_db"，数据会保存到磁盘
)
# add_texts: 将文本批量写入 Chroma
#   内部流程：调用 embedding_function.embed_documents(texts) → 得到向量 → 存入 Chroma
#   返回值: List[str]  每个文本块对应的唯一 ID 列表
doc_ids = chroma_store.add_texts(texts=chunk_texts)
print(f"已索引 {len(doc_ids)} 个文档块到 Chroma 向量数据库。")


# ========================= 2. Retrieval & Generation（检索与生成阶段） =========================
# 目标：根据用户问题，从向量数据库检索相关文档，让 LLM 基于这些文档生成回答。

# ----- Step 5. Setup LLM: 初始化大语言模型 -----
llm = ChatZhipuAI(
    # model: 智谱的对话模型名称
    #   "glm-4":      GLM-4 旗舰模型，综合能力强，适合复杂任务
    #   "glm-4-flash": 轻量快速版，响应更快、成本更低，适合简单任务
    #   "glm-4-air":   性价比版
    model="glm-4",

    # temperature: 生成文本的"随机性/创造性"（取值范围 0.0 ~ 1.0）
    #   0.0: 每次都选择概率最高的词，回答确定性最强，几乎固定不变
    #   0.2: 有一定灵活性但不偏离上下文，适合 RAG（需要忠于检索到的知识）
    #   0.8: 创意性强，回答多样，适合写诗、脑暴等场景
    #   1.0: 最大随机性，可能产生不可预测的内容
    #   为什么 RAG 用低 temperature？
    #     RAG 的目标是基于检索到的文档准确回答问题，不需要"创意"，
    #     低 temperature 能确保 LLM 忠实引用上下文，减少编造。
    temperature=0.2,

    # 其他常用参数（本例使用默认值）：
    #   top_p:        核采样阈值 (0~1)，和 temperature 二选一
    #   max_tokens:   单次最大输出 token 数，不设置则自动
    #   api_key:      手动指定 API Key，不传则从环境变量 ZHIPUAI_API_KEY 读取
)

# ----- Step 6. Retrieve: 创建检索器 -----
# as_retriever(): 将向量数据库包装成"检索器"对象
#   检索器的核心方法是 .invoke(query) / .get_relevant_documents(query)
#   内部流程：embed_query(query) → 在向量空间中找最近邻 → 返回 Top-K 个最相似的 Document
retriever = chroma_store.as_retriever(
    # 常用参数（本例使用默认值）：
    #   search_type:  检索策略类型
    #     "similarity"        默认，纯向量相似度
    #     "mmr"               Max Marginal Relevance，兼顾相似度和多样性，避免重复结果
    #     "similarity_score_threshold"  只返回相似度高于阈值的文档
    #   search_kwargs: 传递给检索器的额外参数
    #     {"k": 4}           返回最相似的 4 个文档（默认值）
    #     {"score_threshold": 0.5}  配合 similarity_score_threshold 使用
    #     {"fetch_k": 20}    配合 mmr 使用，先取 20 个候选再排多样性
)

# ----- Step 7. Augment: 准备 Prompt 模板 -----
# 为什么需要 Prompt 模板？
#   RAG 的核心是把"检索到的上下文"和"用户问题"组合成一个完整的提示词，
#   告诉 LLM："请只根据下面这些资料回答问题，不知道就说不知道"。
#   这能有效减少 LLM 凭空编造（幻觉）。
langsmith_client = Client()  # LangSmith 客户端，用于从 Hub 拉取共享的 prompt 模板
prompt = langsmith_client.pull_prompt(
    # "rlm/rag-prompt": LangChain Hub 上的标准 RAG prompt 名称
    #   模板大致是：使用以下上下文回答问题，如果不知道就说不知道。
    #   上下文：{context}
    #   问题：{question}
    "rlm/rag-prompt",

    # dangerously_pull_public_prompt: 允许拉取公开的 prompt
    #   因为 "rlm/rag-prompt" 是公开模板，需要显式声明此参数
    dangerously_pull_public_prompt=True,
)

# ----- Step 8. Generate: 构建 RAG 链 & 生成回答 -----
# 下面使用 LCEL（LangChain Expression Language）管道式写法：
#   | 是 LCEL 的核心操作符，读作"然后"，将前一步的输出作为后一步的输入。

rag_chain = (
    # 第一部分：准备输入字典 {"context": ..., "question": ...}
    #   retriever | RunnableLambda(format_docs) 的意思是：
    #     用 retriever 检索到相关文档 → 用 format_docs 拼接成字符串
    #   为什么要用 RunnableLambda 包装？
    #     format_docs 是一个普通 Python 函数，IDE 静态类型检查会警告类型不匹配；
    #     RunnableLambda 是 LangChain 提供的"函数转 Runnable"包装器，
    #     既能消除 IDE 警告，又能和 LCEL 管道写法完美融合。
    #   RunnablePassthrough() 的意思是：
    #     原样透传用户输入的 query 字符串，不做任何处理
    #   最终结果是一个字典，正好匹配 prompt 模板的两个占位符
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}

    # | prompt: 将字典填入 prompt 模板，生成完整的提示词文本
    | prompt

    # | llm: 将提示词发给 GLM-4，获取 LLM 的原始响应对象（AIMessage）
    | llm

    # | StrOutputParser(): 从 AIMessage 中提取纯文本字符串
    #   去掉响应对象中的元数据，只保留生成的回答文字
    | StrOutputParser()
)

# invoke(): 执行链，传入输入，获取最终输出
query = "What is Task Decomposition?请用中文回答！"
# query = "你认识王红博吗？"
response = rag_chain.invoke(query)
print(f"\n问题：{query}")
print(f"回答：\n{response}")

# ----- Step 9. Cleanup（可选）: 清理向量数据库 -----
# 删除整个集合，释放资源。学习阶段可以先注释掉，方便反复测试。
# chroma_store.delete_collection()
