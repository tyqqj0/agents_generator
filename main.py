import os
import asyncio
from langchain_openai import ChatOpenAI
from simple_agent_framework import ChatAgent, ToolAgent, RouterAgent
from dotenv import load_dotenv

# 将.env导入环境变量
load_dotenv()

# 检查环境变量中是否设置了OpenAI API密钥
base_url = "https://chatapi.zjt66.top/v1"
key_name = "ANTHROPIC_API_KEY"
api_key = os.environ.get(key_name)
if not api_key:
    raise ValueError(f"请设置{key_name}环境变量")


async def main():
    # 创建一个大语言模型实例，使用OpenAI的gpt-3.5-turbo模型
    # temperature参数控制输出的随机性，值越低越确定性，值越高越创造性
    llm = ChatOpenAI(model="claude-3-5-haiku-20241022", temperature=0.7, base_url=base_url, api_key=api_key)

    # 创建通用聊天代理
    # ChatAgent是simple_agent_framework中的基础代理类型，用于一般对话
    chat_agent = ChatAgent(
        name="通用助手",  # 代理的名称
        model=llm,  # 使用的语言模型
        system_prompt="你是一个有用的AI助手，提供清晰准确的回答，并在必要时解释复杂概念。保持友好和专业的语气。",  # 系统提示，定义代理的行为
    )

    # MCP服务器配置 - GitHub
    # 这是用于GitHub工具代理的模型上下文协议服务器配置
    # Model Context Protocol (MCP)允许代理使用外部工具
    github_config = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
    }

    # 创建GitHub工具代理
    # ToolAgent是能够使用外部工具的代理，这里配置为使用GitHub API
    github_agent = ToolAgent(
        name="github助手",  # 代理名称
        model=llm,  # 使用的语言模型
        mcp_config=github_config,  # MCP服务器配置
        system_prompt="你是GitHub专家，可以帮助用户查询仓库信息、问题和拉取请求。",  # 系统提示
    )

    # 创建旅游专家代理
    # 专门用于回答旅游相关问题的聊天代理
    travel_agent = ChatAgent(
        name="旅游助手",
        model=llm,
        system_prompt="你是一个旅游专家，提供有关目的地、住宿和活动的详细建议。",
    )

    # 创建编程专家代理
    # 专门用于回答编程问题的聊天代理
    coding_agent = ChatAgent(
        name="编程助手",
        model=llm,
        system_prompt="你是一个编程专家，帮助用户编写、调试和理解代码。",
    )

    # 创建路由代理
    # RouterAgent可以根据用户输入，自动将请求路由到最合适的专业代理
    router = RouterAgent(
        name="路由器",
        model=llm,
        agents={
            "通用": chat_agent,  # 通用问题
            "github": github_agent,  # GitHub相关查询
            "旅游": travel_agent,  # 旅游咨询
            "编程": coding_agent,  # 编程问题
        },
    )

    # 测试聊天代理的功能
    print("===== 测试聊天代理 =====")
    response = await chat_agent.agenerate("人工智能的未来发展趋势是什么？")
    print(f"回复: {response.content}\n")

    # 测试路由代理的功能
    # 通过不同类型的问题展示路由代理如何分发请求
    print("===== 测试路由代理 =====")
    queries = [
        "法国的首都是什么?",  # 通用知识问题
        "我计划去日本旅行。东京有什么值得看的?",  # 旅游相关问题
        "如何在Python中写一个递归函数?",  # 编程问题
        "帮我查询langchain仓库的信息",  # GitHub相关查询
    ]

    # 循环测试不同类型的查询
    for query in queries:
        print(f"\n查询: {query}")
        response = await router.agenerate(query)
        print(f"路由到: {response.metadata.get('routed_to')}")  # 显示路由到了哪个代理
        print(f"回复: {response.content}")  # 显示代理的回复内容


# 当脚本直接运行时执行main函数
if __name__ == "__main__":
    asyncio.run(main())
