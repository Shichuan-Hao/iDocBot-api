"""
统一的 .env 配置文件加载器。
所有模块 import env_config 即可获取配置值。
"""
import os
from pathlib import Path


# 加载 .env 文件
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_PARSED = {}

if _ENV_FILE.exists():
    with open(_ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                # 去掉首尾引号
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                # 展开 ${VAR} 引用（先展开系统变量，再展开已解析的值）
                for ref_key, ref_val in os.environ.items():
                    val = val.replace(f"${{{ref_key}}}", ref_val)
                for ref_key, ref_val in _PARSED.items():
                    val = val.replace(f"${{{ref_key}}}", ref_val)
                _PARSED[key] = val
                os.environ[key] = val


# -------- 便捷接口 --------
def get_str(key: str, default: str = "") -> str:
    return _PARSED.get(key, os.environ.get(key, default))


def get_bool(key: str, default: bool = False) -> bool:
    val = get_str(key, str(default))
    return val.lower() in ("true", "1", "yes", "on")


def get_list(key: str, default: list = None) -> list:
    val = get_str(key)
    if not val:
        return default or []
    return [item.strip() for item in val.split(",") if item.strip()]
