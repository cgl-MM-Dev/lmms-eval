import os
import json
from typing import Dict, Optional

from .base import ServerInterface
from .factory import ProviderFactory
from .protocol import Request, Response, ServerConfig
from .utils import JudgePromptBuilder, ResponseParser


def get_server(server_name: str, config: ServerConfig = None) -> ServerInterface:
    """
    Get a server instance by name.

    Args:
        server_name: Name of the server to instantiate.
        config: Optional configuration for the server.

    Returns:
        An instance of ServerInterface.
    """
    return ProviderFactory.create_provider(api_type=server_name, config=config)

_JUDGE_MODEL_INSTANCES: Dict[str, ServerInterface] = {}

def get_judge_model(judge_name: str = "default") -> Optional[ServerInterface]:
    """获取配置中对应的LLM Judge服务端实例，支持单模型或多模型配置。"""
    global _JUDGE_MODEL_INSTANCES
    if judge_name in _JUDGE_MODEL_INSTANCES:
        return _JUDGE_MODEL_INSTANCES[judge_name]

    judge_config_str = os.environ.get("LLM_JUDGE_CONFIG")
    if not judge_config_str:
        return None
    
    try:
        config_dict = json.loads(judge_config_str)
    except json.JSONDecodeError:
        return None
    
    # 解析配置，兼容单层和嵌套字典
    judge_cfg = None
    if judge_name in config_dict and isinstance(config_dict[judge_name], dict):
        judge_cfg = config_dict[judge_name]
    elif "model_name" in config_dict:
        # backward compatibility for single flat config
        judge_cfg = config_dict
    
    if not judge_cfg:
        return None

    # Configure environment variables for API key and base URL if provided
    api_type = judge_cfg.get("api_type", "openai")
    
    if api_type in ["openai", "async_openai"]:
        if "api_key" in judge_cfg:
            os.environ["OPENAI_API_KEY"] = judge_cfg["api_key"]
        if "base_url" in judge_cfg:
            os.environ["OPENAI_API_URL"] = judge_cfg["base_url"]
            
    server_config = ServerConfig(
        model_name=judge_cfg.get("model_name", ""),
        temperature=judge_cfg.get("temperature", 0.0),
        max_tokens=judge_cfg.get("max_tokens", 1024),
        top_p=judge_cfg.get("top_p", None)
    )
    
    instance = get_server(api_type, config=server_config)
    _JUDGE_MODEL_INSTANCES[judge_name] = instance
    return instance


__all__ = [
    "ServerInterface",
    "ServerConfig",
    "Request",
    "Response",
    "ProviderFactory",
    "JudgePromptBuilder",
    "ResponseParser",
    "get_server",
    "get_judge_model",
]
