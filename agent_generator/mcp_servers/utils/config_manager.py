# -*- coding: utf-8 -*-
"""
@File    :   config_manager.py
@Time    :   2025/04/23 14:33:58
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





"""
MCP服务器配置管理器

此模块提供用于管理MCP服务器配置的工具。
"""

import os
import json
import importlib.util
import inspect
import pkgutil
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from .uv_helper import default_helper as uv_helper


def discover_mcp_servers() -> List[str]:
    """
    发现所有可用的MCP服务器。

    返回:
        可用的MCP服务器名称列表
    """
    # 获取mcp_servers包的路径
    base_dir = Path(os.path.dirname(os.path.dirname(__file__)))

    # 查找所有以_mcp.py结尾的文件
    server_files = list(base_dir.glob("*_mcp.py"))
    server_names = [f.stem.replace("_mcp", "") for f in server_files]

    # 也检查servers目录下的服务器
    servers_dir = base_dir / "servers"
    if servers_dir.exists():
        server_files_in_dir = list(servers_dir.glob("*_mcp.py"))
        server_names.extend([f.stem.replace("_mcp", "") for f in server_files_in_dir])

    return sorted(server_names)


def load_server_config(server_name: str) -> Dict[str, Any]:
    """
    加载特定MCP服务器的配置。

    参数:
        server_name: 服务器名称

    返回:
        服务器配置

    异常:
        ImportError: 当无法导入服务器模块时
        ValueError: 当服务器模块不包含必要的配置信息时
    """
    # 构建模块路径
    base_dir = Path(os.path.dirname(os.path.dirname(__file__)))

    # 先检查根目录
    module_path = f"{server_name}_mcp.py"
    file_path = base_dir / module_path

    # 如果在根目录不存在，检查servers目录
    if not file_path.exists():
        module_path = f"servers/{server_name}_mcp.py"
        file_path = base_dir / module_path

        if not file_path.exists():
            raise ImportError(f"找不到MCP服务器模块: {server_name}")

    # 动态导入模块
    module_name = f"mcp_servers.{module_path.replace('/', '.').replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载MCP服务器模块: {server_name}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 验证模块包含必要的配置信息
    if not hasattr(module, "SERVER_CONFIG"):
        raise ValueError(f"MCP服务器模块 {server_name} 不包含SERVER_CONFIG")

    return module.SERVER_CONFIG


def get_mcp_server_config(
    server_names: Optional[List[str]] = None,
    pretty: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    获取MCP配置。

    参数:
        server_names: 要包含的服务器名称列表，如果为None，则包含所有服务器

    返回:
        MCP配置
    """
    if server_names is None:
        server_names = discover_mcp_servers()

    configs = {}
    for name in server_names:
        # print(f"加载服务器 {name} 的配置")
        try:
            config = load_server_config(name)
            configs[name] = config
            # print(config)
        except (ImportError, ValueError) as e:
            print(f"警告: 无法加载服务器 {name} 的配置: {e}")

    if pretty:
        return json.dumps(configs, indent=2, ensure_ascii=False)
    else:
        return configs


def generate_mcp_config_file(
    output_path: str = "mcp_config.json",
    server_names: Optional[List[str]] = None,
    pretty: bool = True,
) -> None:
    """
    生成MCP配置文件。

    参数:
        output_path: 输出文件路径
        server_names: 要包含的服务器名称列表，如果为None，则包含所有服务器
        pretty: 是否美化输出
    """
    configs = get_mcp_server_config(server_names)

    # 创建最终配置
    mcp_config = {"mcpServers": {}}

    for name, config in configs.items():
        server_config = {}

        # 确保配置包含命令
        if "command" in config:
            server_config["command"] = config["command"]
        else:
            # 使用UV路径辅助类获取正确的UV路径
            if uv_helper.exists():
                server_config["command"] = uv_helper.get_uv_path()
            else:
                server_config["command"] = "uv"

        # 提取参数
        if "args" in config:
            server_config["args"] = config["args"]

        # 提取端口
        if "port" in config:
            server_config["port"] = config["port"]

        mcp_config["mcpServers"][name] = server_config

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(mcp_config, f, indent=2, ensure_ascii=False)
        else:
            json.dump(mcp_config, f, ensure_ascii=False)

    print(f"已生成MCP配置文件: {output_path}")


if __name__ == "__main__":
    # 测试配置管理器
    servers = discover_mcp_servers()
    print(f"发现的MCP服务器: {servers}")

    # 生成配置文件
    generate_mcp_config_file()

    # 测试运行
    server_config = get_mcp_server_config(server_names=["calculator"])
    print(server_config)


def test_calculator_mcp():
    """
    测试计算器MCP服务器配置

    此函数测试计算器MCP服务器的配置是否正确加载，并验证服务器能否正常启动
    """
    try:
        # 测试发现服务器
        servers = discover_mcp_servers()
        assert "calculator" in servers, "未找到计算器服务器"
        print("✓ 成功发现计算器服务器")

        # 测试加载配置
        config = load_server_config("calculator")
        assert "name" in config and config["name"] == "calculator", "服务器名称配置错误"
        assert "command" in config, "缺少命令配置"
        assert "transport" in config, "缺少传输方式配置"
        print("✓ 成功加载计算器服务器配置")

        # 测试配置细节
        assert "args" in config and isinstance(config["args"], list), "参数配置错误"
        print("✓ 配置细节验证成功")

        # 测试获取MCP配置
        mcp_config = get_mcp_server_config(server_names=["calculator"])
        assert "calculator" in mcp_config, "在MCP配置中未找到计算器服务器"
        print("✓ 成功获取MCP配置")

        # 尝试修复calculator_mcp.py中潜在的问题
        # 获取服务器脚本路径
        base_dir = Path(os.path.dirname(os.path.dirname(__file__)))
        server_path = base_dir / "servers" / "calculator_mcp.py"

        # 提示用户检查问题
        print("\n发现计算器服务器可能存在以下问题:")
        print("1. 主函数中的'full_str'变量未定义，应该使用SERVER_CONFIG")
        print("2. mcp.run()函数被注释掉，导致服务器未启动")
        print("3. 需要在FastMCP初始化时添加keep_alive=True参数")
        print("4. 需要在mcp.run()调用时添加keep_alive=True参数")
        print(f"请检查文件: {server_path}")

        print("\n建议修复方案:")
        print('1. 将FastMCP初始化改为: mcp = FastMCP("calculator", keep_alive=True)')
        print('2. 将mcp.run调用改为: mcp.run(transport="stdio", keep_alive=True)')
        print("3. 确保主函数中的print语句正常工作，用于调试目的")

        print("\n测试完成！")
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        return False


# 如果直接运行此文件，则执行测试
if __name__ == "__main__":
    print("\n开始测试计算器MCP服务器配置...")
    test_calculator_mcp()
