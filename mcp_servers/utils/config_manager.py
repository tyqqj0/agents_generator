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
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


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


def get_mcp_config(
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
        try:
            config = load_server_config(name)
            configs[name] = config
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
    configs = get_mcp_config(server_names)
    
    # 创建最终配置
    mcp_config = {"mcpServers": {}}

    for name, config in configs.items():
        server_config = {}
        
        # 确保配置包含命令
        if "command" in config:
            server_config["command"] = config["command"]
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
