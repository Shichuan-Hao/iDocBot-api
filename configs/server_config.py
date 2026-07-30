import sys
from configs.env_config import get_str


LLM_DEVICE = "auto"
HTTPX_DEFAULT_TIMEOUT = 300.0

# ---------- 服务器绑定地址 ----------
DEFAULT_BIND_HOST = get_str("BIND_HOST", "0.0.0.0" if sys.platform != "win32" else "127.0.0.1")

# ---------- 主 API 服务 ----------
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": int(get_str("API_SERVER_PORT", "8000")),
}

# ---------- FastChat Controller ----------
FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": int(get_str("CONTROLLER_PORT", "20001")),
    "dispatch_method": "shortest_queue",
}

# ---------- FastChat OpenAI 兼容 API ----------
FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": int(get_str("OPENAI_API_PORT", "20000")),
}

# ---------- Model Worker 配置 ----------
# 可在此处为特定模型设置端口 / device 等
FSCHAT_MODEL_WORKERS = {
    "default": {
        "host": DEFAULT_BIND_HOST,
        "port": 20002,
        "device": LLM_DEVICE,
    },
    # ---- 在线模型只需指定端口 ----
    "zhipu-api": {
        "port": 21001,
    },
}
# Ollama 模型 worker 端口（自动递增分配，避免冲突）
_ollama_base_port = int(get_str("OLLAMA_WORKER_PORT", "21002"))
# 自动为 LLM_MODELS 中的 Ollama 模型加上 worker 端口配置
from configs.model_config import LLM_MODELS, _ollama_models
for _i, _m in enumerate(LLM_MODELS):
    if _m in _ollama_models:
        # 每个 Ollama 模型独立占用一个端口
        FSCHAT_MODEL_WORKERS[_m] = {"port": _ollama_base_port + _i}
