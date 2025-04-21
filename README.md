# Simple Agent Framework

A lightweight, flexible framework for building and using different types of AI agents with unified interfaces. The framework makes it easy to create agents that can use tools through the Model Context Protocol (MCP).

## Features

- **Unified Interface**: Call different types of agents with the same simple interface
- **Quick Agent Creation**: Easily build different types of specialized agents
- **MCP Integration**: Seamlessly use tools via MCP servers
- **Flexible Design**: Simple yet expandable architecture

## Installation

```bash
git clone https://github.com/yourusername/simple_agent_framework.git
cd simple_agent_framework
pip install -e .
```

## Basic Usage

### Chat Agent

```python
from simple_agent_framework import ChatAgent
from langchain_openai import ChatOpenAI

# Create an LLM
llm = ChatOpenAI(model="gpt-4")

# Create a chat agent
chat_agent = ChatAgent(
    name="general_assistant",
    model=llm,
    system_prompt="You are a friendly assistant that specializes in explaining complex concepts."
)

# Use the agent
response = chat_agent.generate("Explain quantum computing in simple terms")
print(response.content)

# Or use the simplest calling method
answer = chat_agent("What is machine learning?")
print(answer)
```

### Tool Agent with MCP

```python
from simple_agent_framework import ToolAgent
from langchain_openai import ChatOpenAI

# Create an LLM
llm = ChatOpenAI(model="gpt-4")

# MCP server configuration
github_config = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"]
}

# Create a tool agent
github_agent = ToolAgent(
    name="github_assistant",
    model=llm,
    mcp_config=github_config,
    system_prompt="You are a GitHub expert who can help users query repository information."
)

# Use the agent
response = github_agent.generate("Check the latest 5 commits in the langchain repository")
print(response.content)
```

### Router Agent

```python
from simple_agent_framework import RouterAgent, ChatAgent, ToolAgent
from langchain_openai import ChatOpenAI

# Create an LLM
llm = ChatOpenAI(model="gpt-4")

# Create specialized agents
chat_agent = ChatAgent(name="chat", model=llm)

web_config = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-brave-search"]
}
web_agent = ToolAgent(name="web", model=llm, mcp_config=web_config)

# Create a router agent
router = RouterAgent(
    name="router",
    model=llm,
    agents={"chat": chat_agent, "web": web_agent}
)

# The router will automatically select the appropriate agent
result = router("Search for the latest news about AI regulation")
print(result)
```

## Extending the Framework

### Creating a Custom Agent

```python
from simple_agent_framework import Agent, AgentResponse
from typing import List, Optional
from langchain_core.messages import BaseMessage

class MyCustomAgent(Agent):
    async def agenerate(self, 
                      query: str, 
                      messages: Optional[List[BaseMessage]] = None,
                      **kwargs) -> AgentResponse:
        # Your custom logic here
        return AgentResponse(
            content="This is a response from my custom agent",
            metadata={"agent_name": self.name}
        )
```

## Requirements

- Python 3.8+
- langchain and langchain-core
- pydantic
- mcp

## License

MIT
