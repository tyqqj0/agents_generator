"""
多MCP服务器工具示例 - 简化版

测试MCP服务器补丁效果
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# 把上级目录添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import ToolAgent
from mcp_servers import get_available_servers, get_mcp_config

# 加载环境变量
load_dotenv()

# 使用适当的API密钥
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("BASE_URL")  # 可选的API base URL


async def test_mcp_connection():
    """仅测试MCP服务器连接"""
    # 获取MCP服务器列表和配置
    names = get_available_servers()
    print(f"可用的MCP服务器: {names}")
    mcp_servers = get_mcp_config(names)

    # 导入所需的类
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # 创建客户端并连接
    try:
        async with MultiServerMCPClient(mcp_servers) as client:
            # 测试get_tools方法
            if hasattr(client, "get_tools"):
                print("客户端有get_tools方法")
                tools = await client.get_tools()
                print(f"获取的工具数量: {len(tools)}")
                for tool in tools:
                    print(f" - {tool.name}: {tool.description}")

            # 测试list_tools方法
            if hasattr(client, "list_tools"):
                print("客户端有list_tools方法")
                try:
                    tools = await client.list_tools()
                    print(f"list_tools返回数量: {len(tools)}")
                except Exception as e:
                    print(f"list_tools调用失败: {e}")

        # 给事件循环额外时间清理
        await asyncio.sleep(0.5)

        print("测试完成")

    except Exception as e:
        print(f"MCP连接测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Windows 需要显式终止子进程
        import psutil

        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            child.kill()
        current = psutil.Process()
        print(f"残留子进程: {current.children(recursive=True)}")
    return True


if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
