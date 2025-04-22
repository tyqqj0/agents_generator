"""
Tool Agent Example

This example demonstrates how to use the ToolAgent with MCP for GitHub interaction.
"""

import os
import asyncio
import logging
from langchain_openai import ChatOpenAI
from agents import ToolAgent

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
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "transport": "stdio",  # 明确指定transport参数
    }

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
            logger.info("代理初始化完成，开始查询")

            try:
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
                logger.error(f"查询过程中出错: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"初始化代理时出错: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"运行主程序时出错: {e}", exc_info=True)
