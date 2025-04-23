# -*- coding: utf-8 -*-
"""
@File    :   __init__.py
@Time    :   2025/04/23 14:35:21
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





"""
Agent Generator

一个用于创建和管理各种智能代理的包
"""

# 导出核心代理组件
from .agents import (
    BaseAgent,
    ToolAgent,
    ReactAgent
)

from .mcp_servers import (
    discover_mcp_servers,
    get_mcp_config,
    generate_mcp_config_file,
    get_available_servers
)

__version__ = "0.1.0"
