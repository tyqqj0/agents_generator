"""
多MCP服务器工具示例

本示例演示如何使用multiple MCP服务器作为工具来创建一个工具代理。
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 把上级目录添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import ToolAgent
from mcp_servers import get_available_servers, get_mcp_config

# 加载环境变量
load_dotenv()

# 使用适当的API密钥
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("BASE_URL")  # 可选的API base URL


async def main():
    # 创建LLM
    llm_args = {
        "model": "claude-3-5-haiku-20241022",  # 替换为你可用的模型
        "temperature": 0.7,
    }

    # 如果设置了base_url和api_key，添加它们
    if base_url:
        llm_args["base_url"] = base_url
    if api_key:
        llm_args["api_key"] = api_key

    llm = ChatOpenAI(**llm_args)

    # 设置MCP服务器配置 - 一次使用多个MCP服务器
    names = get_available_servers()
    print(f"可用的MCP服务器: {names}")
    mcp_servers = get_mcp_config(names)
    print(f"MCP服务器配置: {mcp_servers}")

    # 方法1: 使用上下文管理器
    print("\n===== 方法1: 使用上下文管理器 =====")
    async with ToolAgent(
        name="多工具助手",
        model=llm,
        mcp_servers=mcp_servers,
        system_prompt="你是一个助手，可以同时访问计算器和天气工具。",
    ) as multi_tool_agent:
        # 测试mcp_servers
        mcp_server_query = "请列出所有可用的MCP服务器"
        print(f"\n查询: {mcp_server_query}")
        mcp_server_response = await multi_tool_agent.agenerate(mcp_server_query)
        print(f"回复: {mcp_server_response.content}")

        # 测试计算器功能
        calc_query = "计算54除以9是多少，然后求结果的平方根"
        print(f"\n查询: {calc_query}")
        calc_response = await multi_tool_agent.agenerate(calc_query)
        print(f"回复: {calc_response.content}")

        # 测试天气功能
        weather_query = "告诉我北京和上海的天气情况"
        print(f"\n查询: {weather_query}")
        weather_response = await multi_tool_agent.agenerate(weather_query)
        print(f"回复: {weather_response.content}")

        # # 测试同时使用多个工具
        # combined_query = "计算25的平方根，然后比较这个数值与上海当前的温度"
        # print(f"\n查询: {combined_query}")
        # combined_response = await multi_tool_agent.agenerate(combined_query)
        # print(f"回复: {combined_response.content}")


if __name__ == "__main__":
    asyncio.run(main())
