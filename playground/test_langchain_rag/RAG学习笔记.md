# RAG 入门实践：用 LangChain + GLM-4 + Chroma 搭建一个智能问答系统

> 学习目标：理解 RAG（检索增强生成）的核心流程，并用 Python 实现一个可运行的最小 Demo。
> 技术栈：LangChain、智谱 GLM-4、Chroma 向量数据库、智谱 Embedding-2。

---

## 一、什么是 RAG？

**RAG = Retrieval（检索） + Augmented（增强） + Generation（生成）**

简单来说，RAG 就是让大语言模型（LLM）在回答问题之前，先“查资料”，再基于查到的资料来回答。这样可以让模型：

1. **减少幻觉**：不凭空编造，回答有依据。
2. **获取新知识**：回答训练数据之外的内容，比如公司内部文档、最新博客文章。
3. **可追溯**：回答来源于哪些文档片段，一目了然。

下面这张图展示了一个最经典的 RAG 流程：

```
┌────────────────────────────────────────────────────────────┐
│ 1. Indexing（索引阶段）                                      │
│    Documents → Chunking → Document chunks → Vectorize      │
│    → Vector Database                                       │
│                                                            │
│ 2. Retrieval & Generation（检索与生成阶段）                  │
│    User Query → Vectorize & Search → Retrieve              │
│    → Augment（Prompt + Relevant Contexts + Query）         │
│    → LLM Generate → Response                               │
└────────────────────────────────────────────────────────────┘
```

**两个阶段的核心理解：**

- **Indexing（离线阶段）**：先把知识处理、向量化，存进向量数据库。只做一次，之后可以反复查询。
- **Retrieval & Generation（在线阶段）**：用户提问时，实时检索相关知识，再让 LLM 生成回答。

---

## 二、RAG 的两种形态

### 形态 1：基础 RAG（Naive RAG）

对应第一张流程图，流程非常直接：

1. 把文档切分成小块
2. 把每个小块变成向量，存进向量数据库
3. 用户提问时，把问题也变成向量
4. 在向量数据库里找最相似的文档块
5. 把文档块和用户问题一起塞进 Prompt
6. LLM 根据 Prompt 生成回答

**优点**：简单、好理解、容易实现。  
**缺点**：
- 如果问题表述和文档中的表述不一致，可能检索不到相关内容（语义鸿沟）。
- 如果文档块太大，检索不精准；太小，信息不完整。
- 如果问题本身需要拆解，直接检索可能效果不佳。

### 形态 2：高级 RAG（Advanced RAG）

第二张图展示了一个更复杂的流程：

```
Documents
  → Chunking（带 chunk size limit）
  → Vector Store（+ Tagging 元数据标签）

Query: "How to Make my Grandmother's Chocolate Cake?"
  → Specialized Prompt
  → LLM Transform（改写/扩展/生成关键词查询）
  → Keyworded Query
  → Vector Search
  → 检索到 Chunk 2、Chunk 3
  → LLM Transform（结合 Specialized Prompt 生成回答）
  → Response
```

**相比基础 RAG，多了这些优化：**

| 优化点 | 作用 |
|--------|------|
| **Chunk Size Limit** | 严格控制每个块的大小，平衡信息完整性和检索精度 |
| **Tagging** | 给文档块打标签（如来源、类别），支持按标签过滤，提高检索质量 |
| **Query Transformation** | 让 LLM 先把用户问题改写成更适合检索的"关键词查询"，解决语义鸿沟 |
| **Specialized Prompt** | 针对改写和生成阶段分别设计不同的 Prompt，让每一步都更专业 |

**总结**：基础 RAG 适合入门和理解核心思想；高级 RAG 通过查询改写、元数据过滤、重排序等技术解决实际生产中的各种问题。

---

## 三、代码实现：从流程图到 Python

本项目的代码文件：`langchain_rag.py`

下面按照 RAG 的两个阶段，逐段讲解代码。

### 3.1 环境准备

```python
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

load_dotenv()  # 从 .env 读取 ZHIPUAI_API_KEY
```

**依赖安装：**

```bash
pip install --upgrade langchain langchain-community langchainhub \
    httpx httpx-sse PyJWT langchain-chroma bs4 python-dotenv
```

---

### 3.2 阶段一：Indexing（索引）

#### Step 1：加载文档（Load Documents）

```python
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()
```

- `web_paths`：要抓取的网页 URL。
- `bs_kwargs.parse_only`：用 BeautifulSoup 的 `SoupStrainer` 只抓取正文、标题等区域，过滤导航栏等无关内容。

#### Step 2：切分文档（Chunking）

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)
```

- `chunk_size=1000`：每个文本块最多 1000 个字符。
- `chunk_overlap=200`：相邻块重叠 200 个字符，防止关键句子被切断。

**为什么需要重叠？**  
比如一句话 `"苹果公司的总部位于加州库比蒂诺"`，如果没有重叠，可能被切成 `"苹果公司的总部位于"` 和 `"加州库比蒂诺"` 两块，语义就断了。重叠可以保证关键上下文不被破坏。

#### Step 3 & 4：Embedding + 存入向量数据库

```python
class EmbeddingGenerator:
    def __init__(self, model_name="embedding-2"):
        self.model_name = model_name
        self.client = ZhipuAI()

    def embed_documents(self, texts):
        # 批量把文档块转成向量
        ...

    def embed_query(self, query):
        # 把用户查询转成向量
        ...

embedding_generator = EmbeddingGenerator(model_name="embedding-2")
chunk_texts = [chunk.page_content for chunk in chunks]

chroma_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_generator,
    create_collection_if_not_exists=True,
)
doc_ids = chroma_store.add_texts(texts=chunk_texts)
```

- `EmbeddingGenerator`：自定义的嵌入模型包装器，必须实现 `embed_documents` 和 `embed_query` 两个方法，才能被 LangChain 使用。
- `embedding-2`：智谱的 Embedding 模型，把文本转成 1024 维向量。
- `Chroma`：开源向量数据库，负责存储向量并提供相似度搜索。
- `collection_name`：集合名，类似关系数据库中的"表"。
- `add_texts`：把文本写入 Chroma，内部会自动调用 `embed_documents` 生成向量。

**索引阶段总结：** 网页 → 文本 → 小块 → 向量 → 向量数据库。

---

### 3.3 阶段二：Retrieval & Generation（检索与生成）

#### Step 5：初始化大模型

```python
llm = ChatZhipuAI(
    model="glm-4",
    temperature=0.2,
)
```

- `model="glm-4"`：使用智谱 GLM-4 作为生成模型。
- `temperature=0.2`：控制输出的随机性。RAG 场景需要稳定、基于上下文的回答，所以用较低的值。

#### Step 6：创建检索器

```python
retriever = chroma_store.as_retriever()
```

把向量数据库转换成"检索器"。默认使用相似度搜索，返回最相关的 `k` 个文档块（默认 `k=4`）。

#### Step 7：准备 Prompt 模板

```python
langsmith_client = Client()
prompt = langsmith_client.pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True,
)
```

从 LangChain Hub 拉取一个标准的 RAG Prompt 模板，大致内容是：

```text
根据以下上下文回答问题。如果不知道答案，请说不知道。

上下文：
{context}

问题：{question}

答案：
```

这样可以把检索到的文档内容填充到 `{context}`，用户问题填充到 `{question}`。

#### Step 8：构建 RAG 链并生成回答

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "What is Task Decomposition?"
response = rag_chain.invoke(query)
print(response)
```

**链式流程解析：**

1. `retriever | RunnableLambda(format_docs)`：用问题检索相关文档块，再用 `format_docs` 把多个块拼接成一个字符串。
2. `RunnablePassthrough()`：原样透传用户的问题。
3. 组合成字典 `{"context": "...", "question": "..."}`，填入 Prompt 模板。
4. `prompt | llm | StrOutputParser()`：把完整 Prompt 发给 GLM-4，并把模型输出解析成纯文本。

**为什么用 `RunnableLambda(format_docs)`？**  
因为 `format_docs` 是普通 Python 函数，直接写 `retriever | format_docs` 在 IDE 里会有类型警告。`RunnableLambda` 是 LangChain 提供的包装器，可以把普通函数转成 `Runnable`，消除警告且不影响运行结果。

---

## 四、关键参数速查表

| 参数 | 所在位置 | 含义 | 建议 |
|------|----------|------|------|
| `web_paths` | `WebBaseLoader` | 要抓取的网页 URL | 可以填多个 URL |
| `parse_only` | `bs_kwargs` | 只提取指定 HTML class 的内容 | 用于过滤无关信息 |
| `chunk_size` | `RecursiveCharacterTextSplitter` | 每个文本块最大字符数 | 1000~2000 较常用 |
| `chunk_overlap` | `RecursiveCharacterTextSplitter` | 相邻块重叠字符数 | chunk_size 的 10%~20% |
| `model_name` | `EmbeddingGenerator` | 嵌入模型名称 | `embedding-2` 适合入门 |
| `collection_name` | `Chroma` | 向量数据库集合名 | 按项目命名 |
| `embedding_function` | `Chroma` | 嵌入模型实例 | 必须实现两个方法 |
| `temperature` | `ChatZhipuAI` | 生成随机性 | RAG 建议 0.1~0.3 |
| `k` | `as_retriever(search_kwargs=...)` | 返回最相关的文档数 | 默认 4，可按需调整 |
| `search_type` | `as_retriever` | 检索策略 | `similarity` / `mmr` |

---

## 五、学习心得

1. **RAG 不是替代 LLM，而是给 LLM 配了一个"外接大脑"**  
   LLM 本身知识有限且固定，RAG 让它能动态查阅最新、最专业的资料。

2. **Chunking 是 RAG 的关键调优点**  
   切分太大，检索不精准；切分太小，上下文丢失。需要根据实际文档类型反复调试 `chunk_size` 和 `chunk_overlap`。

3. **Embedding 模型决定检索质量**  
   同样的文档和查询，不同 Embedding 模型的检索效果可能差别很大。入门可以用免费的 `embedding-2`，生产环境需要测试多个模型。

4. **Prompt 工程不可忽视**  
   一个清晰的 RAG Prompt（"根据上下文回答，不知道就说不知道"）能显著减少幻觉。

5. **从基础 RAG 到高级 RAG 有很多优化空间**  
   比如查询改写（Query Transformation）、重排序（Re-ranking）、元数据过滤（Tagging）、混合检索（向量 + 关键词）等，都是实际项目中常用的进阶技术。

---

## 六、延伸阅读

- LangChain 官方文档：https://python.langchain.com/
- 智谱 AI 开放平台：https://open.bigmodel.cn/
- 文本切分可视化工具：https://chunkviz.up.railway.app/
- RAG 经典论文：《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》

---

## 七、RAG 最新进展关注渠道

RAG 领域发展极快（几乎每周都有新论文、新框架发布），持续跟踪是保持技术嗅觉的关键。以下是我整理的优质渠道：

### 论文 & 学术前沿

| 渠道 | 说明 |
|------|------|
| **arXiv CS.IR / CS.CL** | RAG 最新论文的首发地。关键词搜 `RAG`、`Retrieval-Augmented`、`Dense Retrieval` |
| | 🔗 https://arxiv.org/list/cs.IR/recent |
| **Semantic Scholar** | 学术搜索引擎，支持按论文引用关系追踪，可设置关键词订阅邮件 |
| | 🔗 https://www.semanticscholar.org/ |
| **Papers With Code** | 论文 + 开源实现，按任务/数据集浏览，可看 SOTA 排行 |
| | 🔗 https://paperswithcode.com/task/retrieval-augmented-generation |

### 中文社区 & 公众号（推荐优先关注）

| 渠道 | 说明 |
|------|------|
| **宝玉的科技周报** | 每周汇总 AI 前沿资讯，含 RAG/Agent 最新论文解读，质量很高 |
| **量子位** | 国内 AI 资讯头部账号，RAG/Agent 相关动态覆盖及时 |
| **夕小瑶科技说** | NLP/RAG 方向的深度技术解读，论文精读系列做得很扎实 |
| **机器学习算法与自然语言处理** | 论文解读 + 技术实践，新人友好 |
| **GitHub Trending** | 每周看热门的 RAG/LLM 开源项目 |
| | 🔗 https://github.com/trending/python?since=weekly |

### 技术博客 & Newsletter

| 渠道 | 说明 |
|------|------|
| **LangChain Blog** | RAG 最佳实践、LangChain 新特性、官方教程 |
| | 🔗 https://blog.langchain.dev/ |
| **LlamaIndex Blog** | 专注数据框架下的 RAG 方案，含大量实战案例和 cookbook |
| | 🔗 https://www.llamaindex.ai/blog |
| **Lilian Weng's Blog** | OpenAI 研究员，RAG/Agent 系列文章是经典入门必读 |
| | 🔗 https://lilianweng.github.io/ |
| **The Batch (Andrew Ng)** | Andrew Ng 团队周报，每周精选 3~5 篇 AI 进展总结 |
| | 🔗 https://www.deeplearning.ai/the-batch/ |

### 值得关注的 GitHub 项目

| 项目 | 说明 |
|------|------|
| **LangChain** | RAG 开发最主流框架 |
| 🔗 https://github.com/langchain-ai/langchain |
| **LlamaIndex** | 数据为中心的 RAG 框架，和 LangChain 互补 |
| 🔗 https://github.com/run-llama/llama_index |
| **ragas** | RAG 评估框架，衡量检索质量和生成质量 |
| 🔗 https://github.com/explodinggradients/ragas |
| **anything-llm** | 开箱即用的 RAG 应用，适合参考其实现方案 |
| 🔗 https://github.com/Mintplex-Labs/anything-llm |
| **Dify / FastGPT** | 国内成熟的 RAG 应用平台，源码值得学习 |
| 🔗 https://github.com/langgenius/dify / https://github.com/labring/FastGPT |
| **RAG 论文合集** | awesome-RAG 系列，持续更新 RAG 方向所有重要论文和项目 |
| 🔗 https://github.com/hymie122/RAG-Survey |

### 实战建议

1. **每周花 30 分钟**：扫一眼 arXiv CS.IR 最新论文标题 + Semantic Scholar 订阅邮件
2. **读 1~2 篇高质量博客**：LangChain / LlamaIndex 官方博客更新频率适中，每篇都值得精读
3. **收藏 1~2 个优质 GitHub 项目**：跟踪他们的 Release Notes，看看主流框架的迭代方向
4. **关注公众号聚合**：中文公众号更新快，通勤时间刷一刷，知道最近在讨论什么就够了

---

## 八、名词解释

> 按代码中出现的频率和重要性排序，帮助初学者扫清概念障碍。

| 名词 | 一句话解释 | 通俗理解 |
|------|-----------|----------|
| **LLM** | Large Language Model，大语言模型 | 就是 ChatGPT / GLM-4 这类"超级大脑"，能理解、生成人类语言 |
| **RAG** | Retrieval-Augmented Generation，检索增强生成 | 让 LLM 先查资料再回答，而不是凭记忆瞎编 |
| **Embedding** | 嵌入 / 向量化 | 把一段文字变成一串数字（如 1024 个浮点数），语义相近的文本数字也相近 |
| **Vector** | 向量 | 就是那一串数字本身。比如 `[0.12, -0.34, 0.78, ...]` |
| **Vector Database** | 向量数据库 | 专门存"向量"的数据库，能快速找到和你的查询向量最相似的那些向量。类比：传统数据库按字段精确匹配，向量数据库按"语义相似度"搜索 |
| **Chroma** | 一个轻量级开源向量数据库 | 本文使用的向量数据库，支持内存/磁盘存储，适合学习和中小规模项目 |
| **Document** | LangChain 中的文档对象 | 包含两部分的抽象：`page_content`（文本内容）+ `metadata`（来源 URL、页码等元数据） |
| **Chunking** | 文档切分 / 分块 | 把一篇长文章切成多个小片段，方便检索和塞进 LLM 上下文窗口 |
| **chunk_size** | 每个文档块的最大字符数 | 太小信息不完整，太大检索不精准 |
| **chunk_overlap** | 相邻块之间的重叠字符数 | 防止一句话被切断在两个块的边界 |
| **Prompt** | 提示词 | 你给 LLM 的"指令"。RAG 的 Prompt 通常长这样：_请根据以下资料回答问题：{资料}，问题：{问题}_ |
| **Prompt Template** | 提示词模板 | 带有占位符 `{context}`、`{question}` 的 Prompt 骨架，使用时填入具体内容 |
| **Retriever** | 检索器 | 根据用户查询，从向量数据库中找到最相关的 K 个文档块 |
| **k** | Top-K，返回最相关结果的数量 | `k=4` 表示返回最相似的 4 个文档块 |
| **Similarity** | 相似度 | 两个向量在空间中距离的度量，距离越近语义越相关 |
| **MMR** | Max Marginal Relevance | 一种检索策略，在"相似度"和"多样性"之间平衡，避免返回内容重复的文档块 |
| **Temperature** | 温度参数（0~1） | 控制 LLM 输出的随机性：0 = 确定但呆板，1 = 创意但可能跑偏 |
| **Hallucination** | 幻觉 | LLM 一本正经地编造不存在的事实。是 RAG 想要解决的核心问题 |
| **LCEL** | LangChain Expression Language | LangChain 的链式调用语法，用 `\|` 管道符串联多个处理步骤 |
| **RunnablePassthrough** | 透传器 | LCEL 中的一个组件，不做任何处理，原样把输入传给下一步 |
| **RunnableLambda** | 函数包装器 | 把普通 Python 函数包成 LangChain 的 `Runnable` 对象，使其能用于 `\|` 管道 |
| **StrOutputParser** | 字符串输出解析器 | 把 LLM 返回的 AIMessage 对象中的元数据去掉，只保留纯文本回答 |
| **BeautifulSoup (bs4)** | HTML 解析库 | 用来从网页中提取指定区域的文字内容，过滤掉广告和导航栏 |
| **.env / dotenv** | 环境变量管理 | 把 API Key 等敏感信息放在 `.env` 文件中，用 `load_dotenv()` 加载，避免硬编码到代码里 |
| **LangSmith** | LangChain 配套的可观测平台 | 可以追踪 LLM 调用链路、调试 Prompt、评估效果。本文用它从 Hub 拉取 Prompt 模板 |

### 核心概念关系图

```
                    RAG 流程中的名词关系

  原始数据           →  Chunking（文档切分）
  (Documents)               ↓
                      Document Chunks（文档块）
                            ↓
                     Embedding（向量化）
                            ↓
                     Vector Database（向量数据库）
                     (Chroma)
                            ↓  ← 用户 Query（问题）
                      Retriever（检索器）
                       相似度搜索 top-k
                            ↓
                    Relevant Contexts（相关上下文）
                            ↓
                      Prompt Template（提示词模板）
                      填入 context + question
                            ↓
                         LLM（GLM-4）
                            ↓
                        Response（回答）
```

---

*本笔记基于 `langchain_rag.py` 代码和两张 RAG 流程图整理而成，适合初学者理解 RAG 的整体架构和代码实现。*
