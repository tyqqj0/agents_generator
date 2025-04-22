from .agents.base import BaseAgent, AgentResponse
from .agents.chat_agent import ChatAgent
from .agents.tool_agent import ToolAgent
from .agents.router_agent import RouterAgent
from .templates.prompts import (
    DEFAULT_CHAT_PROMPT,
    DEFAULT_TOOL_PROMPT,
    DEFAULT_RESEARCH_PROMPT,
    DEFAULT_CODE_PROMPT,
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
        "AgentResponse",
        "ChatAgent",
        "ToolAgent",
        "RouterAgent",
        "DEFAULT_CHAT_PROMPT",
        "DEFAULT_TOOL_PROMPT",
        "DEFAULT_RESEARCH_PROMPT",
        "DEFAULT_CODE_PROMPT",
        "MultiServerMCPClient",
        "load_mcp_tools",
        "create_mcp_client",
    ]
except ImportError:
    # 如果未安装langchain_mcp_adapters，则仅导出核心组件
    __all__ = [
        "Agent",
        "AgentResponse",
        "ChatAgent",
        "ToolAgent",
        "RouterAgent",
        "DEFAULT_CHAT_PROMPT",
        "DEFAULT_TOOL_PROMPT",
        "DEFAULT_RESEARCH_PROMPT",
        "DEFAULT_CODE_PROMPT",
    ]

__version__ = "0.1.0"
