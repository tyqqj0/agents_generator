# Simple Agent Framework

这是一个轻量级、灵活的框架，用于构建和使用具有统一接口的各种AI代理。该框架使得创建能够通过模型上下文协议（MCP）使用工具的代理变得简单。

## 项目目标

- **提供统一接口**：使用相同的简单接口调用不同类型的代理。
- **简化代理创建**：轻松构建不同类型的专业代理。
- **集成MCP**：通过官方的langchain_mcp_adapters无缝使用MCP服务器工具。
- **灵活设计**：提供一个简单但可扩展的架构。

## 项目结构

```
.                       # 项目根目录
├── .env                # 环境变量 (需要用户自行创建)
├── .venv/              # Python虚拟环境 (建议)
├── examples/           # 使用框架的示例代码
│   ├── chat_agent_example.py
│   ├── custom_tools_agent_example.py
│   ├── router_agent_example.py
│   └── tool_agent_example.py
├── mcp_servers/        # 自定义MCP服务器实现
│   ├── calculator_mcp.py
│   ├── weather_mcp.py
│   ├── requirements.txt
│   └── README.md
├── simple_agent_framework/ # 框架核心代码
│   ├── agents/           # 代理实现
│   │   ├── __init__.py
│   │   ├── base.py         # 代理基类
│   │   ├── chat_agent.py   # 聊天代理
│   │   ├── router_agent.py # 路由代理
│   │   └── tool_agent.py   # 工具代理
│   ├── connectors/       # 连接器实现
│   │   ├── __init__.py
│   │   └── mcp.py        # MCP连接器
│   ├── templates/        # 提示模板
│   │   ├── __init__.py
│   │   └── prompts.py
│   ├── utils/            # 工具函数
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── __init__.py       # 包初始化
├── .gitignore          # Git忽略文件
├── main.py             # 主测试和演示脚本
├── pyproject.toml      # 项目元数据和构建配置
├── README.md           # 本文档
└── setup.py            # 安装脚本
```

## 各组件功能

- **simple_agent_framework/**: 框架的核心包。
    - `agents/`: 包含不同类型的代理实现。
        - `base.py`: 定义了所有代理共享的基础接口 (`Agent`) 和响应格式 (`AgentResponse`)。
        - `chat_agent.py`: 简单的聊天代理，与LLM进行对话。
        - `tool_agent.py`: 可以通过MCP连接器使用外部工具的代理。
        - `router_agent.py`: 根据用户查询将请求路由到最合适的专业代理。
    - `connectors/`: 用于连接外部服务或协议。
        - `mcp.py`: `MCPConnector`类，负责与MCP服务器通信，列出工具和执行工具。
    - `templates/`: 包含预定义的提示模板。
        - `prompts.py`: 为不同类型的代理提供默认的系统提示。
    - `utils/`: 辅助函数。
        - `helpers.py`: 例如，加载不同提供商的LLM。
- **examples/**: 包含如何使用框架中各种代理的示例脚本。
- **mcp_servers/**: 包含自定义的MCP服务器实现（计算器和天气）。
    - 这些服务器是独立的Python脚本，可以单独运行。
    - `README.md`: 提供了关于如何运行和扩展这些服务器的详细说明。
- **main.py**: 一个集成的测试脚本，演示了如何创建和使用各种代理（包括使用自定义MCP服务器的代理），并提供了一个交互式菜单来选择测试模式。
- **setup.py / pyproject.toml**: 用于包的安装和依赖管理。



## 基本用法

框架的核心是 `Agent` 类及其子类。所有代理都提供一致的接口：

- `agent.generate(query)`: 同步生成响应内容（字符串）。
- `agent.agenerate(query)`: 异步生成 `AgentResponse` 对象。
- `agent(query)`: 便捷方法，等同于 `agent.generate(query).content`。



## 扩展框架

### 创建自定义代理

继承 `Agent` 基类并实现 `agenerate` 方法：

```python
from simple_agent_framework import Agent, AgentResponse
from typing import List, Optional
from langchain_core.messages import BaseMessage

class MyCustomAgent(Agent):
    async def agenerate(self, 
                      query: str, 
                      messages: Optional[List[BaseMessage]] = None,
                      **kwargs) -> AgentResponse:
        # 在这里实现你的自定义逻辑
        processed_content = f"Custom agent received: {query}"
        return AgentResponse(
            content=processed_content,
            metadata={"agent_name": self.name, "processed": True}
        )
```

### 创建自定义MCP服务器

参考 `mcp_servers/README.md` 中的说明和示例代码 (`calculator_mcp.py`, `weather_mcp.py`) 来创建你自己的Python MCP服务器。


## 许可证

MIT License
