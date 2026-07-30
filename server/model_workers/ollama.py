import json
from typing import Dict, Iterator, List

from fastchat.conversation import Conversation

from configs.env_config import get_str
from server.model_workers.base import (
    ApiChatParams,
    ApiEmbeddingsParams,
    ApiModelWorker,
)
from server.utils import get_httpx_client
from configs import logger, log_verbose
from fastchat import conversation as conv


class OllamaWorker(ApiModelWorker):
    """
    通过 Ollama 的 OpenAI 兼容接口（默认 http://<host>:11434/v1）调用本地或远程 Ollama 服务。
    """

    DEFAULT_EMBED_MODEL = "bge-m3"

    def __init__(
        self,
        *,
        model_names: List[str] = ("ollama",),
        controller_addr: str = None,
        worker_addr: str = None,
        version: str = "qwen2.5:7b",
        **kwargs,
    ):
        kwargs.update(
            model_names=model_names,
            controller_addr=controller_addr,
            worker_addr=worker_addr,
        )
        kwargs.setdefault("context_len", 4096)
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Iterator[Dict]:
        """
        调用 Ollama 的 OpenAI 兼容 chat completions 接口，支持 stream 输出。
        """
        params.load_config(self.model_names[0])
        if log_verbose:
            logger.info(f"{self.__class__.__name__}:params: {params}")

        # 从配置中拿到 api_base_url（默认指向 Ollama）
        api_base_url = (
            params.api_base_url
            or get_str("OLLAMA_API_BASE", "http://192.168.1.9:11434/v1")
        )
        # 去掉末尾斜杠，避免出现 //chat/completions
        api_base_url = api_base_url.rstrip("/")
        url = f"{api_base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            # Ollama 默认不校验 key，填一个占位符即可
            "Authorization": f"Bearer {params.api_key or 'ollama'}",
        }

        data = {
            "model": params.version or self.version,
            "messages": params.messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "stream": True,
        }

        text = ""
        with get_httpx_client() as client:
            try:
                with client.stream("POST", url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.strip() or "[DONE]" in line:
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        try:
                            resp = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if choices := resp.get("choices"):
                            delta = choices[0].get("delta", {}) or {}
                            chunk = delta.get("content")
                            if chunk:
                                text += chunk
                                yield {"error_code": 0, "text": text}
                        else:
                            logger.error(
                                f"请求 Ollama API 时返回异常：{resp}"
                            )
            except Exception as e:
                logger.error(f"调用 Ollama 接口失败：{e}")
                yield {"error_code": 500, "text": f"调用 Ollama 失败：{e}"}

    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        """
        调用 Ollama 的 OpenAI 兼容 /v1/embeddings 接口。
        """
        params.load_config(self.model_names[0])
        embed_model = params.embed_model or self.DEFAULT_EMBED_MODEL

        api_base_url = (
            params.api_base_url
            or get_str("OLLAMA_API_BASE", "http://192.168.1.9:11434/v1")
        )
        api_base_url = api_base_url.rstrip("/")
        url = f"{api_base_url}/embeddings"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {params.api_key or 'ollama'}",
        }

        result = []
        with get_httpx_client() as client:
            for text in params.texts:
                data = {"model": embed_model, "input": text}
                try:
                    resp = client.post(url, headers=headers, json=data)
                    resp.raise_for_status()
                    emb = resp.json()["data"][0]["embedding"]
                    result.append(emb)
                except Exception as e:
                    logger.error(
                        f"Ollama embeddings 请求失败（model={embed_model}）：{e}"
                    )
                    return {"code": 500, "msg": str(e)}

        return {"code": 200, "data": result}

    def make_conv_template(
        self, conv_template: str = None, model_path: str = None
    ) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个乐于助人的助手。",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n###",
            stop_str="###",
        )


if __name__ == "__main__":
    import sys
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    worker = OllamaWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21002",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21002)