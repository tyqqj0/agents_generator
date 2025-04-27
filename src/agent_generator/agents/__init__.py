# -*- coding: utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2025/04/23 14:33:36
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""






"""
Agent模块包

提供各种类型的智能代理实现
"""

from .base import BaseAgent
from .null_agent import NullAgent
from .tool_agent import ToolAgent
from .react_agent import ReactAgent
from .critic_agent import CriticAgent
from .autoact_agent import AutoActAgent
# 注释掉不存在的导入
# from .agents.chat_agent import ChatAgent
# from .agents.router_agent import RouterAgent
from .templates.prompts import (
    DEFAULT_TOOL_PROMPT,
    DEFAULT_CRITIC_SYSTEM_PROMPT,
    DEFAULT_CRITIQUE_PROMPT,
    DEFAULT_CORRECTION_PROMPT,
)
# 导出MCP相关功能
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    # 创建一个简单的同步版MCP客户端管理器函数
    def create_mcp_client(mcp_servers_config):
        """创建一个MCP客户端管理器

        Args:
            mcp_servers_config: 多MCP服务器配置字典
                格式: {"server_name": {"command": "...", "args": [...], "transport": "..."}}

        Returns:
            MultiServerMCPClient 实例，可用于获取工具
        """
        return MultiServerMCPClient(mcp_servers_config)

    __all__ = [
        "BaseAgent",
        "ToolAgent",
        "ReactAgent",
        "CriticAgent",
        "NullAgent",
        "AutoActAgent",
        "MultiServerMCPClient",
        "load_mcp_tools",
        "create_mcp_client",
        "DEFAULT_TOOL_PROMPT",
        "DEFAULT_CRITIC_SYSTEM_PROMPT",
        "DEFAULT_CRITIQUE_PROMPT",
        "DEFAULT_CORRECTION_PROMPT",
    ]
except ImportError:
    # 如果未安装langchain_mcp_adapters，则仅导出核心组件
    __all__ = [
        "BaseAgent",
        "ToolAgent",
        "ReactAgent",
        "CriticAgent",
        "NullAgent",
        "AutoActAgent",
        "DEFAULT_TOOL_PROMPT",
        "DEFAULT_CRITIC_SYSTEM_PROMPT",
        "DEFAULT_CRITIQUE_PROMPT",
        "DEFAULT_CORRECTION_PROMPT",
    ]
    
    
def get_available_agents():
    return [
        "ToolAgent",
        "ReactAgent",
        "CriticAgent",
        "AutoActAgent",
    ]

# 包版本信息
__version__ = "0.1.0"
