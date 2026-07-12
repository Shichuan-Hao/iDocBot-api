#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import Body, HTTPException
from typing import List, Union, Optional

# 使用LangChain调用ChatGLM3-6B的依赖包
from langchain.chains.llm import LLMChain
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

# 日志包
from loguru import logger


def chat(
    query: str = Body("", description="用户的输入"),
    model_name: str = Body("glm4-9b-chat", description="基座模型的名称"),
    temperature: float= Body(0.8, description="大模型参数：采样温度", ge=0.0, le=2.0),
    max_tokens: Optional[int] = Body(None, description="大模型参数： 最大输入 Token 限制")
):
    """
        :param query: 用户输入的问题
        :param model_name: 使用哪个大模型作为后端服务
        :param temperature: 采样温度
        :param max_tokens: 最大输入Token限制
        :return:  大模型的回复结果
    """

    logger.info("Received query: {}", query)
    logger.info("Model name: {}", model_name)
    logger.info("Temperature: {}", temperature)
    logger.info("Max tokens: {}", max_tokens)
