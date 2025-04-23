"""
计算器MCP服务器

一个使用Python实现的简单计算器MCP服务器，提供基本的数学计算功能。
"""

import math
import os
import sys
from mcp.server.fastmcp import FastMCP

# 将上一级文件夹添加到sys.path，确保可以找到uv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 初始化 FastMCP server
mcp = FastMCP("calculator", keep_alive=True)

# 服务器配置信息
SERVER_NAME = "calculator"
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
    "description": "基础计算器MCP服务器",
}


@mcp.tool(name="add", description="将两个数字相加")
async def add(a: float, b: float) -> float:
    """将两个数字相加。

    Args:
        a: 第一个数字
        b: 第二个数字

    Returns:
        两个数字的和
    """
    return a + b


@mcp.tool(name="subtract", description="从第一个数字中减去第二个数字")
async def subtract(a: float, b: float) -> float:
    """从第一个数字中减去第二个数字。

    Args:
        a: 被减数
        b: 减数

    Returns:
        两个数字的差
    """
    return a - b


@mcp.tool(name="multiply", description="将两个数字相乘")
async def multiply(a: float, b: float) -> float:
    """将两个数字相乘。

    Args:
        a: 第一个数字
        b: 第二个数字

    Returns:
        两个数字的积
    """
    return a * b


@mcp.tool(name="divide", description="将第一个数字除以第二个数字")
async def divide(a: float, b: float) -> float:
    """将第一个数字除以第二个数字。

    Args:
        a: 被除数
        b: 除数

    Returns:
        两个数字的商

    Raises:
        ValueError: 当除数为0时
    """
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


@mcp.tool(name="power", description="计算第一个数字的第二个数字次幂")
async def power(base: float, exponent: float) -> float:
    """计算基数的指数次幂。

    Args:
        base: 底数
        exponent: 指数

    Returns:
        底数的指数次幂
    """
    return math.pow(base, exponent)


@mcp.tool(name="sqrt", description="计算一个数字的平方根")
async def sqrt(number: float) -> float:
    """计算一个数字的平方根。

    Args:
        number: 要计算平方根的数字

    Returns:
        数字的平方根

    Raises:
        ValueError: 当输入为负数时
    """
    if number < 0:
        raise ValueError("不能计算负数的平方根")
    return math.sqrt(number)


if __name__ == "__main__":
    # 启动MCP服务器
    # print(f"启动MCP服务器: {SERVER_NAME}")
    # full_str = {
    #     "name": SERVER_NAME,
    #     "command": "uv",
    #     "args": [
    #         "--directory",
    #         os.path.dirname(os.path.abspath(__file__)),
    #         "run",
    #         os.path.basename(__file__),
    #     ],
    #     "transport": "stdio",
    #     "description": "基础计算器MCP服务器",
    # }
    # print(SERVER_CONFIG)
    mcp.run(transport="stdio")
