# -*- coding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2025/04/23 14:35:34
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





from agent_generator.agents import ToolAgent, ReactAgent, CriticAgent
from agent_generator.mcp_servers import get_available_servers, get_mcp_config

import asyncio
import dotenv
import os
from langchain_openai import ChatOpenAI
import json


dotenv.load_dotenv()


api_name = "OPENAI_API_KEY"
api_key = os.environ.get(api_name)
base_url = os.environ.get("BASE_URL")
model = "gpt-4o-mini"




async def tool_agent_example():
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    agent = ToolAgent(name="mcp_agent", model=llm, mcp_servers=get_mcp_config(get_available_servers()))
    async with agent:
        result = await agent.agenerate("你好，请给出北京今天天气")
        print(result["messages"][-1].content)
        
async def react_agent_example():
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    agent = ReactAgent(name="mcp_agent", model=llm, mcp_servers=get_mcp_config(get_available_servers()))
    print(json.dumps(agent.tools, indent=4))
    async with agent:
        result = await agent.agenerate("你好，请给出北京今天天气，并且计算1+626287+北京今天的天气温度")
        print(result["messages"][-1].content)

async def compare_agents(prompt, image_url=None):
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    
    # print("===== ToolAgent 结果 =====")
    # tool_agent = ToolAgent(name="tool_agent", model=llm, mcp_servers=get_mcp_config(["tavily", "code"]))
     
    # async with tool_agent:
    #     # print(tool_agent.tools)
    #     tool_agent.visualize()
    #     result = await tool_agent.agenerate(prompt, image_url=image_url)
    #     print(result["messages"][-1].content)
    
    # print("\n===== ReactAgent 结果 =====")
    # react_agent = ReactAgent(name="react_agent", model=llm, mcp_servers=get_mcp_config(["tavily"]))
    # async with react_agent:
    #     result = await react_agent.agenerate(prompt, image_url=image_url)
    #     print(result["messages"][-1].content)
        
    print("\n===== CriticAgent 结果 =====")
    critic_agent = CriticAgent(name="critic_agent", model=llm, mcp_servers=get_mcp_config(["tavily"]), use_markers=False)
    async with critic_agent:
        # critic_agent.visualize()
        result = await critic_agent.agenerate(prompt, image_url=image_url)
        
        # 获取消息历史(这里的messages已经是清理过的，只包含用户输入和AI回答)
        messages = result.get("messages", [])
        
        # 直接打印整个交互历史
        print("用户提问和AI回答历史:")
        for msg in messages:
            print(f"{msg.type.upper()}: {msg.content}")
        
        # 如果你只想打印最终结果，可以使用:
        # last_ai = next((msg for msg in reversed(messages) if msg.type == "ai"), None)
        # if last_ai:
        #     print(f"最终回答: {last_ai.content}")
        
        # 打印迭代信息
        print(f"\n完成状态: {result.get('is_complete', False)}")
        print(f"迭代次数: {result.get('iterations', 0)}")
        
        # 打印其他可能有用的状态信息
        if 'critiques' in result and result['critiques']:
            print(f"最后批评: {result['critiques'][:100]}...")

if __name__ == "__main__":
    # asyncio.run(tool_agent_example())
    # asyncio.run(react_agent_example())
    asyncio.run(compare_agents(prompt = "请分析这个图片，指出图片中的学校，并且搜索学校的精准地址", image_url="https://image-tyqqj.oss-cn-beijing.aliyuncs.com/Snipaste_2025-04-23_18-27-43.png"))



