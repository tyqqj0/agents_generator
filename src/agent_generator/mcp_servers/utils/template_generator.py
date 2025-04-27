# -*- coding: utf-8 -*-
"""
@File    :   template_generator.py
@Time    :   2025/04/23 14:34:41
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





"""
MCP服务器模板生成器

此模块提供创建MCP服务器模板的工具。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any


def create_mcp_server_template(
    server_name: str,
    tools: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    description: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """
    创建MCP服务器模板文件

    参数:
        server_name: 服务器名称
        tools: 工具配置列表
        output_dir: 输出目录，默认为mcp_servers目录
        description: 服务器描述
        overwrite: 是否覆盖已有文件

    返回:
        生成的文件路径
    """
    if output_dir is None:
        # 默认放在mcp_servers/servers目录下
        base_dir = Path(os.path.dirname(os.path.dirname(__file__)))
        output_dir = str(base_dir / "servers")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 服务器文件名
    filename = f"{server_name}_mcp.py"
    file_path = os.path.join(output_dir, filename)

    # 检查文件是否已存在
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"文件已存在: {file_path}。使用overwrite=True覆盖。")

    # 如果没有提供描述，使用默认描述
    if description is None:
        description = f"{server_name.capitalize()} MCP服务器"

    # 生成服务器文件内容
    content = generate_server_content(server_name, description, tools)

    # 写入文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"已生成MCP服务器模板文件: {file_path}")
    return file_path


def generate_server_content(
    server_name: str, description: str, tools: List[Dict[str, Any]]
) -> str:
    """
    生成服务器文件内容

    参数:
        server_name: 服务器名称
        description: 服务器描述
        tools: 工具配置列表

    返回:
        服务器文件内容
    """
    # 导入语句
    imports = ["import os", "import sys", "from typing import Dict, List, Any"]

    # 检查工具需要的特殊导入
    special_imports = set()
    for tool in tools:
        if tool.get("type") == "math":
            special_imports.add("import math")
        elif tool.get("type") == "datetime":
            special_imports.add("from datetime import datetime, timedelta")
        elif tool.get("type") == "random":
            special_imports.add("import random")

    imports.extend(sorted(special_imports))
    imports.append("from mcp.server.fastmcp import FastMCP")

    # 文件头部
    header = f'''"""
{description}

一个使用Python实现的MCP服务器。
"""

{os.linesep.join(imports)}

# 初始化 FastMCP server
mcp = FastMCP("{server_name}")
'''

    # 生成工具函数
    tool_functions = []
    for tool in tools:
        tool_functions.append(generate_tool_function(tool))

    # 文件尾部
    footer = """

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run(transport="stdio")
"""

    # 组合所有部分
    return header + os.linesep.join(tool_functions) + footer


def generate_tool_function(tool: Dict[str, Any]) -> str:
    """
    生成工具函数代码

    参数:
        tool: 工具配置

    返回:
        工具函数代码
    """
    name = tool.get("name", "tool_name")
    description = tool.get("description", f"执行{name}操作")
    params = tool.get("parameters", [])
    return_type = tool.get("return_type", "Dict[str, Any]")
    return_description = tool.get("return_description", "操作结果")

    # 构建函数签名
    param_list = []
    for param in params:
        param_name = param.get("name", "param")
        param_type = param.get("type", "str")
        param_list.append(f"{param_name}: {param_type}")

    params_str = ", ".join(param_list)

    # 构建函数文档
    docstring_lines = [f'"""{description}', ""]
    if params:
        docstring_lines.append("Args:")
        for param in params:
            param_name = param.get("name", "param")
            param_desc = param.get("description", f"{param_name}参数")
            docstring_lines.append(f"    {param_name}: {param_desc}")
        docstring_lines.append("")

    docstring_lines.append("Returns:")
    docstring_lines.append(f"    {return_description}")
    docstring_lines.append('"""')

    docstring = os.linesep.join(["    " + line for line in docstring_lines])

    # 函数实现
    implementation = tool.get("implementation", "    # TODO: 实现函数逻辑\n    pass")

    # 组合函数
    return f"""

@mcp.tool(name="{name}", description="{description}")
async def {name}({params_str}) -> {return_type}:
{docstring}
{implementation}
"""


def create_calculator_server(
    output_dir: Optional[str] = None, overwrite: bool = False
) -> str:
    """
    创建计算器服务器模板

    参数:
        output_dir: 输出目录
        overwrite: 是否覆盖现有文件

    返回:
        生成的文件路径
    """
    tools = [
        {
            "name": "add",
            "description": "将两个数字相加",
            "parameters": [
                {"name": "a", "type": "float", "description": "第一个数字"},
                {"name": "b", "type": "float", "description": "第二个数字"},
            ],
            "return_type": "float",
            "return_description": "两个数字的和",
            "implementation": "    return a + b",
            "type": "math",
        },
        {
            "name": "subtract",
            "description": "从第一个数字中减去第二个数字",
            "parameters": [
                {"name": "a", "type": "float", "description": "被减数"},
                {"name": "b", "type": "float", "description": "减数"},
            ],
            "return_type": "float",
            "return_description": "两个数字的差",
            "implementation": "    return a - b",
            "type": "math",
        },
        {
            "name": "multiply",
            "description": "将两个数字相乘",
            "parameters": [
                {"name": "a", "type": "float", "description": "第一个数字"},
                {"name": "b", "type": "float", "description": "第二个数字"},
            ],
            "return_type": "float",
            "return_description": "两个数字的积",
            "implementation": "    return a * b",
            "type": "math",
        },
        {
            "name": "divide",
            "description": "将第一个数字除以第二个数字",
            "parameters": [
                {"name": "a", "type": "float", "description": "被除数"},
                {"name": "b", "type": "float", "description": "除数"},
            ],
            "return_type": "float",
            "return_description": "两个数字的商",
            "implementation": """    if b == 0:
        raise ValueError("除数不能为零")
    return a / b""",
            "type": "math",
        },
    ]

    return create_mcp_server_template(
        "calculator", tools, output_dir, "计算器MCP服务器", overwrite
    )


def create_weather_server(
    output_dir: Optional[str] = None, overwrite: bool = False
) -> str:
    """
    创建天气服务器模板

    参数:
        output_dir: 输出目录
        overwrite: 是否覆盖现有文件

    返回:
        生成的文件路径
    """
    tools = [
        {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": [
                {"name": "city", "type": "str", "description": "要查询天气的城市名称"}
            ],
            "return_type": "Dict[str, Any]",
            "return_description": "包含城市天气信息的字典",
            "implementation": """    # 这里应添加实际天气API调用
    # 示例返回虚拟数据
    weather_data = {
        "city": city,
        "temperature": "23°C",
        "condition": "晴朗",
        "humidity": "45%",
        "wind": "东北风 3级",
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return weather_data""",
            "type": "datetime",
        },
        {
            "name": "list_available_cities",
            "description": "列出所有可查询天气的城市",
            "parameters": [],
            "return_type": "List[str]",
            "return_description": "可用城市列表",
            "implementation": """    # 返回支持的城市列表
    return ["北京", "上海", "广州", "深圳", "成都"]""",
            "type": "none",
        },
    ]

    return create_mcp_server_template(
        "weather", tools, output_dir, "天气MCP服务器", overwrite
    )


if __name__ == "__main__":
    # 测试生成器
    create_calculator_server(overwrite=True)
    create_weather_server(overwrite=True)
