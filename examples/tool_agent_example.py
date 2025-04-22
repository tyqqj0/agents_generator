"""
Tool Agent Example

This example demonstrates how to use the ToolAgent with MCP for GitHub interaction.
"""

import os
import asyncio
import logging
import sys
from pathlib import Path
from langchain_openai import ChatOpenAI
from agents import ToolAgent

# 添加项目根目录到Python路径，确保可以导入mcp_servers模块
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

try:
    from mcp_servers.utils.uv_helper import UVPathHelper

    uv_helper = UVPathHelper()
except ImportError:
    print("警告: 无法导入UV路径辅助类，将使用默认命令 'uv'")
    uv_helper = None

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


async def main():
    # Create an LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)

    # MCP server configuration for GitHub
    github_config = {
        "command": (
            uv_helper.get_uv_path() if uv_helper and uv_helper.exists() else "uv"
        ),
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "transport": "stdio",  # 明确指定transport参数
    }

    logger.info(f"使用UV命令: {github_config['command']}")

    try:
        # Create a GitHub agent
        github_agent = ToolAgent(
            name="github_assistant",
            model=llm,
            mcp_config=github_config,
            system_prompt="You are a GitHub expert. Help users find information about repositories, issues, and pull requests.",
        )

        # 使用异步上下文管理器运行代理
        async with github_agent:
            # Use the agent to get information about a repository
            response = await github_agent.agenerate(
                "Find information about the langchain repository. How many stars does it have?"
            )

            logger.info(f"响应: {response.content}")
            logger.info(f"元数据: {response.metadata}")

            # You can also use the synchronous interface
            logger.info("\n第二次查询 (同步):")
            result = github_agent.generate(
                "List the top 3 contributors to the LangGraph repository"
            )
            logger.info(f"响应: {result.content}")
    except Exception as e:
        logger.error(f"执行代理时出错: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"运行主程序时出错: {e}", exc_info=True)
