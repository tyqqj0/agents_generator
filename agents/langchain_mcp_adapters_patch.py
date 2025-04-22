"""
langchain_mcp_adapters 补丁模块

解决MultiServerMCPClient缺少list_tools方法的问题和返回值兼容性问题
"""

import sys
import asyncio
import importlib
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

# 尝试导入原始的MultiServerMCPClient类
# try:
#     from langchain_mcp_adapters.client import (
#         MultiServerMCPClient as OriginalMultiServerMCPClient,
#     )
#     from langchain_mcp_adapters.tools import load_mcp_tools as original_load_mcp_tools

#     # 检查是否需要应用补丁
#     original_get_tools = OriginalMultiServerMCPClient.get_tools

#     # 创建一个扩展的MultiServerMCPClient类，添加list_tools方法
#     class PatchedMultiServerMCPClient(OriginalMultiServerMCPClient):
#         """
#         补丁版MultiServerMCPClient，添加list_tools方法以兼容旧版API
#         """

#         async def list_tools(self):
#             """
#             兼容性方法，调用get_tools方法

#             Returns:
#                 与get_tools方法相同的返回值
#             """
#             return await self.get_tools()

#         async def get_tools(self):
#             """
#             重写get_tools方法，确保返回值是一个awaitable对象
#             """
#             # 调用原始方法
#             result = original_get_tools(self)

#             # 检查返回值是否可等待
#             if asyncio.iscoroutine(result):
#                 # 如果是可等待对象，直接返回
#                 return await result
#             else:
#                 # 如果不是可等待对象，直接返回
#                 return result

#     # 替换原始模块中的类
#     sys.modules["langchain_mcp_adapters.client"].MultiServerMCPClient = (
#         PatchedMultiServerMCPClient
#     )

#     # 创建一个补丁版本的load_mcp_tools函数
#     async def patched_load_mcp_tools(client: Any):
#         """
#         补丁版load_mcp_tools，处理MultiServerMCPClient的兼容性

#         Args:
#             client: MCP客户端对象

#         Returns:
#             工具列表
#         """
#         # 检查客户端类型
#         if hasattr(client, "get_tools"):
#             try:
#                 # 获取工具
#                 tools = client.get_tools()

#                 # 检查返回值是否可等待
#                 if asyncio.iscoroutine(tools):
#                     tools = await tools

#                 return tools
#             except Exception as e:
#                 print(f"警告: get_tools调用失败: {e}")
#                 if hasattr(client, "list_tools"):
#                     try:
#                         return await client.list_tools()
#                     except Exception as e2:
#                         print(f"警告: list_tools调用也失败: {e2}")
#                         raise e2
#                 else:
#                     raise e
#         else:
#             # 否则使用原始函数
#             return await original_load_mcp_tools(client)

#     # 替换原始模块中的函数
#     sys.modules["langchain_mcp_adapters.tools"].load_mcp_tools = patched_load_mcp_tools

#     print("✓ 成功应用langchain_mcp_adapters补丁")

# except ImportError as e:
#     print(f"警告: 无法应用langchain_mcp_adapters补丁: {e}")
# except Exception as e:
#     print(f"警告: 应用langchain_mcp_adapters补丁时出错: {e}")
#     import traceback

#     traceback.print_exc()


# def apply_patch():
#     """
#     显式应用补丁
#     """
#     # 补丁已在导入时应用，此函数仅用于显式调用
#     pass
