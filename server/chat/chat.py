#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from fastapi import Body, HTTPException
from typing import Optional

# 使用 LangChain + OpenAI 兼容接口调用 Ollama 部署的大模型
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 日志包
from loguru import logger


# Ollama 服务地址（OpenAI 兼容接口）
OLLAMA_BASE_URL = "http://192.168.1.9:11434/v1"


def chat(query: str = Body("", description="用户的输入"),
         model_name: str = Body("glm4:9b", description="基座模型的名称"),
         temperature: float = Body(0.8, description="大模型参数：采样温度", ge=0.0, le=2.0),
         max_tokens: Optional[int] = Body(4096, description="大模型参数：最大输出Token限制"),
         ):
    """
    :param query: 用户输入的问题
    :param model_name: 使用哪个大模型作为后端服务（Ollama 模型名，如 glm4:9b）
    :param temperature: 采样温度
    :param max_tokens: 最大输出Token限制
    :return:  大模型的回复结果
    """

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
