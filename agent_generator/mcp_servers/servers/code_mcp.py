"""
代码执行MCP服务器

一个使用Python实现的代码执行MCP服务器，提供安全的代码运行环境。
基于E2B Sandbox实现代码执行功能。
"""

import os
import sys
import json
import logging
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
import logging

# 将上一级文件夹添加到sys.path，确保可以找到uv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
import dotenv
dotenv.load_dotenv()

# 配置日志
if os.environ.get("LOGGING_MODE") == "off":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("e2b-code-mcp-server")

# 初始化 FastMCP server
mcp = FastMCP("code", keep_alive=True)

# 服务器配置信息
SERVER_NAME = "code"
SERVER_CONFIG = {
    "name": SERVER_NAME,
    "command": "uv",
    "args": [
        "--directory",
        os.path.dirname(os.path.abspath(__file__)),
        "run",
        os.path.basename(__file__),
    ],
    "transport": "stdio",
    "description": "代码执行MCP服务器",
}

# 获取E2B API密钥
E2B_API_KEY = os.getenv("E2B_API_KEY")
if E2B_API_KEY:
    logger.info("成功获取E2B API密钥")
else:
    logger.warning("未设置E2B_API_KEY环境变量")

# 尝试导入E2B代码执行环境
try:
    from e2b_code_interpreter import Sandbox
    HAS_E2B = True
except ImportError:
    logger.warning("未安装e2b_code_interpreter，将使用模拟沙箱环境")
    HAS_E2B = False

    # 创建一个模拟的沙箱类用于开发和测试
    class MockSandbox:
        def __init__(self, api_key=None):
            pass

        def run_code(self, code):
            logger.info(f"模拟执行代码: {code}")
            # 简单的执行结果模拟
            class MockExecution:
                def __init__(self, code):
                    self.logs = type('obj', (object,), {
                        'stdout': f"模拟输出: 执行了代码\n{code}",
                        'stderr': ""
                    })
            
            return MockExecution(code)


@mcp.tool(name="run_code", description="在E2B安全沙箱中运行Python代码")
async def run_code(code: str, pure_str_output: bool = False) -> Dict[str, Any]:
    """在安全的沙箱环境中执行Python代码。

    Args:
        code: 要执行的Python代码

    Returns:
        包含代码执行结果的字典，包括标准输出和标准错误

    Raises:
        Exception: 当代码执行出错时
    """
    try:
        logger.info(f"正在执行代码: {code}")
        
        # 根据是否安装了E2B选择沙箱环境
        if HAS_E2B:
            if not E2B_API_KEY:
                return {
                    "stdout": "",
                    "stderr": "未设置E2B_API_KEY环境变量，无法执行代码",
                    "success": False
                }
            sbx = Sandbox(api_key=E2B_API_KEY)
        else:
            sbx = MockSandbox()
            
        # 执行代码
        execution = sbx.run_code(code)
        
        # 构建执行结果
        if pure_str_output:
            stdout = execution.logs.stdout
            stderr = execution.logs.stderr
            # 解析["column1\n", "column2\n", "column3\n"]形式的列表，转化为字符串
            stdout = "".join(stdout)
            stderr = "".join(stderr)
            result = {
                "stdout": stdout,
                "stderr": stderr,
                "success": True
            }
        else:
            result = {
                "stdout": execution.logs.stdout,
                "stderr": execution.logs.stderr,
                "success": True
            }
        
        return result
    except Exception as e:
        logger.error(f"代码执行错误: {str(e)}")
        return {
            "stdout": "",
            "stderr": f"代码执行失败: {str(e)}",
            "success": False
        }


# @mcp.tool(name="run_javascript", description="在E2B安全沙箱中运行JavaScript代码")
# async def run_javascript(code: str) -> Dict[str, Any]:
#     """在安全的沙箱环境中执行JavaScript代码。

#     Args:
#         code: 要执行的JavaScript代码

#     Returns:
#         包含代码执行结果的字典，包括标准输出和标准错误

#     Raises:
#         Exception: 当代码执行出错时
#         NotImplementedError: 当E2B未安装或JavaScript执行未实现时
#     """
#     try:
#         logger.info(f"正在执行JavaScript代码: {code}")
        
#         if not HAS_E2B:
#             return {
#                 "stdout": "",
#                 "stderr": "JavaScript执行功能需要安装E2B库",
#                 "success": False
#             }
            
#         if not E2B_API_KEY:
#             return {
#                 "stdout": "",
#                 "stderr": "未设置E2B_API_KEY环境变量，无法执行JavaScript代码",
#                 "success": False
#             }
            
#         # 注意：这部分代码假设E2B支持JavaScript，实际实现可能需要根据E2B的API调整
#         sbx = Sandbox(api_key=E2B_API_KEY, template="javascript")  # 假设可以通过template参数指定JavaScript环境
#         execution = sbx.run_code(code)
        
#         # 构建执行结果
#         result = {
#             "stdout": execution.logs.stdout,
#             "stderr": execution.logs.stderr,
#             "success": True
#         }
        
#         return result
#     except Exception as e:
#         logger.error(f"JavaScript代码执行错误: {str(e)}")
#         return {
#             "stdout": "",
#             "stderr": f"JavaScript代码执行失败: {str(e)}",
#             "success": False
#         }


if __name__ == "__main__":
    # 启动MCP服务器
    logger.info(f"启动代码执行MCP服务器: {SERVER_NAME}")
    mcp.run(transport="stdio")
