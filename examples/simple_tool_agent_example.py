"""
Simple Tool Agent Example

这个示例展示了如何使用ToolAgent和常见工具，不涉及MCP配置。
"""

import os
import asyncio
import logging
import sys
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from agents import ToolAgent

# 添加项目根目录到Python路径
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

# 尝试导入UV路径辅助类（对于简单示例不是必须的）
try:
    from mcp_servers.utils.uv_helper import UVPathHelper

    uv_helper = UVPathHelper()
    print(f"UV环境路径: {uv_helper.base_dir}")
    print(f"UV环境存在: {uv_helper.exists()}")
except ImportError:
    uv_helper = None
    print("UV路径辅助类不可用，但对于本示例不是必需的")

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load API key from environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请设置OPENAI_API_KEY环境变量")

# 可选：设置Tavily API密钥
tavily_api_key = os.environ.get("TAVILY_API_KEY")


# 创建自定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 要计算的数学表达式，例如 "2 + 2" 或 "sin(30)"

    Returns:
        计算结果
    """
    try:
        # 注意：这种方式在生产环境中不安全，仅用于演示
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {str(e)}"


async def main():
    # 创建语言模型
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    # 准备工具列表
    tools = [calculator]

    # 如果有Tavily API密钥，添加搜索工具
    if tavily_api_key:
        search_tool = TavilySearchResults(max_results=3)
        tools.append(search_tool)

    try:
        # 创建工具代理
        agent = ToolAgent(
            name="math_assistant",
            model=llm,
            tools=tools,
            system_prompt="你是一个擅长数学计算和信息查询的助手。使用提供的工具帮助用户解决问题。",
        )

        # 运行代理
        logger.info("开始执行代理查询...")
        result = agent.generate("计算23乘以45等于多少？")
        logger.info(f"响应: {result.content}")

        if tavily_api_key:
            # 如果可用，使用搜索工具
            logger.info("\n使用搜索工具进行查询:")
            result = agent.generate("谁发明了相对论？简要介绍一下。")
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
