from agent_generator.agents import ToolAgent
from agent_generator.mcp_servers import get_available_servers, get_mcp_config

import asyncio
import dotenv
import os
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


api_name = "OPENAI_API_KEY"
api_key = os.environ.get(api_name)
base_url = os.environ.get("BASE_URL")


llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, base_url=base_url)
agent = ToolAgent(name="mcp_agent", model=llm, mcp_servers=get_mcp_config(get_available_servers()))

async def main():
    async with agent:
        result = await agent.agenerate("你好，请给出北京今天天气")
        print(result["messages"][-1].content)

asyncio.run(main())



