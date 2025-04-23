"""
Agent Generator

一个用于创建和管理各种智能代理的包
"""

# 导出核心代理组件
from .agents import (
    BaseAgent,
    ToolAgent
)

from .mcp_servers import (
    discover_mcp_servers,
    get_mcp_config,
    generate_mcp_config_file,
    get_available_servers
)

__version__ = "0.1.0"
