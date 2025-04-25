# Agent Generator（代理生成器）

这是一个灵活而强大的框架，用于构建和管理各种类型的AI代理。该框架使开发者能够轻松创建能够通过模型上下文协议（MCP）使用工具的智能代理。

## 项目目标

- **统一接口**：提供一致的简单接口来调用不同类型的AI代理
- **快速创建代理**：简化专业代理的构建过程
- **MCP无缝集成**：通过langchain_mcp_adapters库轻松使用MCP服务器工具
- **灵活架构**：提供可扩展的设计，支持多种代理类型和工具

## 项目结构

```
agent_generator/
├── __init__.py              # 包初始化和主要导出
├── agents/                  # 代理实现
│   ├── __init__.py          # 代理包初始化
│   ├── base.py              # 基础代理类（BaseAgent）
│   ├── tool_agent.py        # 工具代理实现
│   ├── null_agent.py        # 空代理实现
│   ├── critic_agent.py      # 评论家代理实现
│   ├── react_agent.py       # ReAct代理实现
│   └── templates/           # 代理提示模板
├── mcp_servers/             # MCP服务器实现
│   ├── __init__.py          # MCP包初始化
│   ├── README.md            # MCP服务器使用说明
│   ├── servers/             # 各种MCP服务器实现
│   ├── utils/               # MCP工具函数
│   ├── test_calculator_client.py # 计算器服务客户端测试
│   ├── test_code_client.py  # 代码服务客户端测试
│   └── test_tavily.py       # Tavily搜索服务测试
└── requirements.txt         # 项目依赖
```

## 主要组件

### 1. 代理模块 (agents/)

该模块包含不同类型的代理实现，所有代理都继承自`BaseAgent`基类：

- **BaseAgent**：所有代理的基类，定义了统一接口和核心功能
- **ToolAgent**：能够使用工具的代理，支持MCP工具的集成
- **ReactAgent**：实现ReAct（推理+行动）范式的代理
- **CriticAgent**：提供评估和批判性思考的代理
- **NullAgent**：简单的空代理实现，用于测试和基础场景

所有代理都支持多模态输入（文本和图像），并提供异步API。

### 2. MCP服务器模块 (mcp_servers/)

提供了多种MCP服务器的实现，可以轻松集成到代理中：

- **天气服务器**：提供城市天气查询功能
- **计算器服务器**：提供数学计算功能
- **Tavily搜索服务器**：提供网络搜索和内容提取功能

每个服务器都是独立的Python应用程序，可以单独启动和使用。

## 基本用法

### 创建和使用工具代理

```python
from agent_generator import ToolAgent
from langchain_openai import ChatOpenAI
import asyncio
import os

# 加载模型
model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# 创建MCP服务器配置
mcp_servers = {
    "calculator": {
        "command": "python",
        "args": ["agent_generator/mcp_servers/servers/calculator_mcp.py"],
        "transport": "stdio"
    }
}

# 初始化代理
agent = ToolAgent(
    name="计算助手",
    model=model,
    mcp_servers=mcp_servers
)

# 异步使用代理
async def main():
    async with agent:  # 使用上下文管理器自动启动和关闭MCP服务器
        response = await agent.agenerate("计算23.5乘以18.7是多少?")
        print(response)

# 运行异步函数
asyncio.run(main())
```

### 多代理工作流

可以组合使用多个代理来创建复杂的工作流：

```python
from agent_generator import ToolAgent, CriticAgent
from langchain_openai import ChatOpenAI
import asyncio
import os

# 加载模型
model = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# 初始化工具代理
tool_agent = ToolAgent(
    name="搜索助手",
    model=model,
    mcp_servers={"tavily": {...}}  # Tavily搜索服务器配置
)

# 初始化评论家代理
critic_agent = CriticAgent(
    name="评论家",
    model=model
)

# 异步工作流
async def workflow(query):
    async with tool_agent, critic_agent:
        # 使用工具代理获取信息
        search_results = await tool_agent.agenerate(f"搜索关于: {query}")
        
        # 使用评论家代理评估结果
        evaluation = await critic_agent.agenerate(f"评估以下信息的准确性和完整性: {search_results}")
        
        return evaluation

# 运行工作流
asyncio.run(workflow("量子计算的最新进展"))
```

## 支持的MCP服务器

框架集成了多种MCP服务器，包括：

### 1. 计算器服务器

提供数学计算功能，包括基本运算、幂运算和平方根计算。

### 2. Tavily搜索服务器

提供互联网搜索功能，需要Tavily API密钥。功能包括：
- 一般网络搜索（tavily_search）
- 新闻搜索（tavily_news_search）
- 网页内容提取（tavily_extract）

### 3. 其他自定义服务器

可以轻松扩展自己的MCP服务器，参考`mcp_servers/README.md`获取详细说明。

## 依赖项

主要依赖包括：
- langchain-openai
- langchain-anthropic
- langchain
- langgraph
- python-dotenv
- anthropic
- openai
- mcp
- fastapi
- uvicorn

## 如何扩展

### 创建自定义代理

继承`BaseAgent`类并实现必要的方法：

```python
from agent_generator import BaseAgent
from langgraph.graph.graph import CompiledGraph

class MyCustomAgent(BaseAgent):
    def __init__(self, name, model, **kwargs):
        super().__init__(name, model, **kwargs)
        # 自定义初始化代码
    
    def _create_agent(self) -> CompiledGraph:
        # 实现代理创建逻辑
        # 返回编译后的图
        ...
```

### 集成自定义MCP服务器

1. 创建新的MCP服务器Python文件
2. 使用FastMCP库实现所需功能
3. 通过`mcp_servers`参数将其配置到代理中

## 许可证

MIT License
