"""
MCP服务器包

包含自定义Model Context Protocol (MCP) 服务器的集合。
"""

import os
import sys
from .utils.config_manager import (
    discover_mcp_servers,
    get_mcp_config,
    generate_mcp_config_file,
)

# 版本
__version__ = "0.1.0"

# 导出公共API
__all__ = [
    "discover_mcp_servers", 
    "get_mcp_config", 
    "generate_mcp_config_file",
    "get_available_servers",
]

# 将此包的路径添加到系统路径中，确保可以直接运行服务器
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.append(package_dir)

# 定义快捷函数
def get_available_servers():
    """
    获取所有可用的MCP服务器名称
    
    返回:
        服务器名称列表
    """
    return discover_mcp_servers()
