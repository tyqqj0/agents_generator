"""
Tool Agent Example

This example demonstrates how to use the ToolAgent with MCP for GitHub interaction.
"""

import os
import asyncio
from langchain_openai import ChatOpenAI
from simple_agent_framework import ToolAgent

# Load API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

async def main():
    # Create an LLM
    llm = ChatOpenAI(
        model="gpt-4", 
        temperature=0.2
    )
    
    # MCP server configuration for GitHub
    github_config = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"]
    }
    
    # Create a GitHub agent
    github_agent = ToolAgent(
        name="github_assistant",
        model=llm,
        mcp_config=github_config,
        system_prompt="You are a GitHub expert. Help users find information about repositories, issues, and pull requests."
    )
    
    # Use the agent to get information about a repository
    response = await github_agent.agenerate(
        "Find information about the langchain repository. How many stars does it have?"
    )
    
    print(f"Response: {response.content}")
    print(f"Metadata: {response.metadata}")
    
    # You can also use the synchronous interface
    print("\nSecond query (synchronous):")
    result = github_agent.generate("List the top 3 contributors to the LangGraph repository")
    print(f"Response: {result.content}")

if __name__ == "__main__":
    asyncio.run(main()) 