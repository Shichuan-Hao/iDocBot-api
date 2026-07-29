# Ollama 调用返回 404 问题记录

## 问题

通过 FastAPI 接口 `/api/chat` 调用 Ollama 部署的 `glm4:9b` 模型时，接口返回 404 错误：

```json
{
    "detail": "Failed with response: <Response [404 Not Found]>"
}
```

## 原因

代码中使用了 `ChatGLM3`（来自 `langchain_community.llms.chatglm3`）来调用 Ollama 服务。`ChatGLM3` 是为 **ChatGLM 原生 API** 设计的，其请求格式和 URL 路径与 Ollama 的 **OpenAI 兼容接口** 不匹配，导致 Ollama 无法识别请求并返回 404。

Ollama 暴露的是 OpenAI 兼容的 API，端点格式为 `http://<host>:11434/v1/chat/completions`，应使用 `ChatOpenAI` 类并设置 `base_url` 指向 Ollama 服务。

## 解决方法

1. 将 `ChatGLM3` 替换为 `ChatOpenAI`（来自 `langchain_openai`）
2. 将 `base_url` 设为 `http://192.168.1.9:11434/v1`（Ollama OpenAI 兼容端点）
3. 添加 `api_key="ollama"`（Ollama 不需要认证，但 `ChatOpenAI` 要求该参数）
4. 返回值改为 `response.content`（`ChatOpenAI` 返回 `AIMessage` 对象，需取 `.content` 属性）
5. 在 `requirements.txt` 中显式添加 `langchain-openai` 依赖

---

## 解决前的代码

**`server/chat/chat.py`：**

```python
from fastapi import Body, HTTPException
from typing import List, Union, Optional

# 使用LangChain调用ChatGLM3-6B的依赖包
from langchain_classic.chains.llm import LLMChain
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

from loguru import logger


def chat(query: str = Body("", description="用户的输入"),
         model_name: str = Body("glm4-9b-chat", description="基座模型的名称"),
         temperature: float = Body(0.8, description="大模型参数：采样温度", ge=0.0, le=2.0),
         max_tokens: Optional[int] = Body(None, description="大模型参数：最大输入Token限制"),
         ):
    logger.info("Received query: {}", query)
    logger.info("Model name: {}", model_name)
    logger.info("Temperature: {}", temperature)
    logger.info("Max tokens: {}", max_tokens)

    try:
        template = """{query}"""
        prompt = PromptTemplate.from_template(template)

        endpoint_url = "http://192.168.1.9:11434/v1/chat/completions"

        llm = ChatGLM3(
            endpoint_url=endpoint_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        llm_chain = prompt | llm
        response = llm_chain.invoke(query)

        if response is None:
            raise ValueError("Received null response from LLM")

        return {"LLM Response": response}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))
```

---

## 解决后的代码

**`server/chat/chat.py`：**

```python
from fastapi import Body, HTTPException
from typing import Optional

# 使用 LangChain + OpenAI 兼容接口调用 Ollama 部署的大模型
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from loguru import logger


# Ollama 服务地址（OpenAI 兼容接口）
OLLAMA_BASE_URL = "http://192.168.1.9:11434/v1"


def chat(query: str = Body("", description="用户的输入"),
         model_name: str = Body("glm4:9b", description="基座模型的名称"),
         temperature: float = Body(0.8, description="大模型参数：采样温度", ge=0.0, le=2.0),
         max_tokens: Optional[int] = Body(4096, description="大模型参数：最大输出Token限制"),
         ):
    logger.info("Received query: {}", query)
    logger.info("Model name: {}", model_name)
    logger.info("Temperature: {}", temperature)
    logger.info("Max tokens: {}", max_tokens)

    try:
        template = """{query}"""
        prompt = PromptTemplate.from_template(template)

        # 通过 OpenAI 兼容接口调用 Ollama 部署的模型
        llm = ChatOpenAI(
            base_url=OLLAMA_BASE_URL,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key="ollama",  # Ollama 不需要真实的 API Key，但 ChatOpenAI 要求此参数
        )

        llm_chain = prompt | llm
        response = llm_chain.invoke(query)

        if response is None or response.content is None:
            raise ValueError("Received null response from LLM")

        return {"LLM Response": response.content}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))
```

**`requirements.txt` 新增依赖：**

```
langchain-openai==1.3.5
```

---

## 关键差异总结

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| LLM 类 | `ChatGLM3` (原生 API) | `ChatOpenAI` (OpenAI 兼容) |
| 端点配置 | `endpoint_url` | `base_url` |
| 端点路径 | `/v1/chat/completions` | `/v1`（自动拼接） |
| 认证 | 无 | `api_key="ollama"` |
| 返回值 | 原始对象 | `response.content` |
| 默认模型名 | `glm4-9b-chat` | `glm4:9b`（Ollama 格式） |
| 默认 max_tokens | `None` | `4096` |
| 未使用导入 | `LLMChain`, `AIMessage` | 已清理 |
