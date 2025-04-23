# -*- coding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2025/04/23 14:35:34
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





from agent_generator.agents import ToolAgent, ReactAgent
from agent_generator.mcp_servers import get_available_servers, get_mcp_config

import asyncio
import dotenv
import os
from langchain_openai import ChatOpenAI
import json


dotenv.load_dotenv()


api_name = "ANTHROPIC_API_KEY"
api_key = os.environ.get(api_name)
base_url = os.environ.get("BASE_URL")
model = "claude-3-5-sonnet-20241022"




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

async def compare_agents(prompt):
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    
    print("===== ToolAgent 结果 =====")
    tool_agent = ToolAgent(name="tool_agent", model=llm, mcp_servers=get_mcp_config(["tavily", "code"]))
    
    async with tool_agent:
        # print(tool_agent.tools)
        result = await tool_agent.agenerate(prompt)
        print(result["messages"][-1].content)
    
    print("\n===== ReactAgent 结果 =====")
    react_agent = ReactAgent(name="react_agent", model=llm, mcp_servers=get_mcp_config(["tavily"]))
    async with react_agent:
        result = await react_agent.agenerate(prompt)
        print(result["messages"][-1].content)

if __name__ == "__main__":
    # asyncio.run(tool_agent_example())
    # asyncio.run(react_agent_example())
    asyncio.run(compare_agents(prompt = "请写一个python代码，计算28511 + 1234567890，并返回结果"))



