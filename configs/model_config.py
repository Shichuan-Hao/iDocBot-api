from configs.env_config import get_str, get_list


TEMPERATURE = 0.8

# ---------- 当前启用的模型（从 .env 读取） ----------
LLM_MODELS = get_list("LLM_MODELS", ["qwen2.5:7b"])

# ---------- 本地模型权重路径（此处暂时保留空壳，便于迁移） ----------
MODEL_ROOT_PATH = ""

MODEL_PATH = {
    "local_model": {
        # 示例： "chatglm3-6b": "/home/00_rag/model/ZhipuAI/chatglm3-6b",
    },
    "embed_model": {
        # 示例： "bge-large-zh-v1.5": "/home/00_rag/model/AI-ModelScope/bge-large-zh-v1___5",
    },
}

# ---------- Ollama 模型的自动配置（从 .env 读取） ----------
# 遍历 OLLAMA_MODELS 列表，自动生成 ONLINE_LLM_MODEL 条目
_ollama_models = get_list("OLLAMA_MODELS", ["qwen2.5:7b"])
_ollama_api_base = get_str("OLLAMA_API_BASE", "http://192.168.1.9:11434/v1")

ONLINE_LLM_MODEL = {
    # ---- 智谱清言（可选） ----
    "zhipu-api": {
        "api_key": get_str("ZHIPU_API_KEY"),
        "version": "glm-4",
        "provider": "ChatGLMWorker",
    },
    # ---- OpenAI（可选） ----
    "openai-api": {
        "model_name": "gpt-4",
        "api_base_url": get_str("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "api_key": get_str("OPENAI_API_KEY"),
        "openai_proxy": "",
    },
}

# 自动为每个 Ollama 模型生成配置
for _model_name in _ollama_models:
    ONLINE_LLM_MODEL[_model_name] = {
        "api_key": "ollama",
        "version": _model_name,
        "provider": "OllamaWorker",
        "api_base_url": _ollama_api_base,
    }

# ---------- Embedding 模型 ----------
EMBEDDING_MODEL = get_str("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DEVICE = "auto"
