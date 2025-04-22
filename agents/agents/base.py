from typing import List, Dict, Any, Optional, Union, Set
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import asyncio
from contextlib import AsyncExitStack

# 新增导入MCP工具相关库
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from abc import ABC, abstractmethod
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph


class AgentResponse(BaseModel):
    """统一的代理响应格式"""

    content: str
    raw_response: Any = None  # 原始响应对象
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """所有代理的基类，提供统一接口"""

    def __init__(
        self,
        name: str,
        model: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
        memory: Optional[BaseChatModel] = None,
        **kwargs,
    ):
        """
        初始化代理

        Args:
            name: 代理名称
            model: 使用的语言模型
            tools: 预定义的工具列表
            mcp_servers: 多MCP服务器配置，格式为 {"server_name": {"command": "...", "args": [...], "transport": "..."}}
            memory: 用于记忆的模型(代支持)
        """
        self.name = name
        self.model = model
        self.tools = tools or []
        self.mcp_servers = mcp_servers or {}
        self.memory = memory
        self.mcp_client = None
        self.agent = self._create_agent()

    def _create_agent(self) -> CompiledGraph:
        """创建代理, 子类必须实现"""
        raise NotImplementedError

    async def agenerate(self, input_text: str, **kwargs) -> AgentResponse:
        """异步生成回复"""
        if not self._initialized and (self.tools or self.mcp_servers):
            await self.ainit()

        response = await self.llm.ainvoke(
            self._get_messages(input_text),
            functions=self._get_tools_for_llm() if self.tools else None,
            temperature=getattr(self, "temperature", 0.7),
            **kwargs,
        )

        # 处理不同类型的响应
        if hasattr(response, "content"):
            # 标准响应格式
            return AgentResponse(content=response.content)
        elif isinstance(response, dict) and "content" in response:
            # 字典格式响应
            return AgentResponse(content=response["content"])
        elif hasattr(response, "get") and callable(response.get):
            # 类字典对象
            content = response.get("content", str(response))
            return AgentResponse(content=content)
        else:
            # 其他类型，转换为字符串
            return AgentResponse(content=str(response))

    def generate(
        self, query: str, messages: Optional[List[BaseMessage]] = None, **kwargs
    ) -> AgentResponse:
        """同步生成响应"""
        # 准备消息
        message_history = messages or []
        message_history.append(HumanMessage(content=query))

        # 执行代理
        response = self.agent.invoke({"messages": message_history})
        return AgentResponse(content=response.content)

    def __call__(self, query: str, **kwargs) -> str:
        """便捷调用方法，直接返回内容字符串"""
        return self.generate(query, **kwargs).content

    async def _setup_mcp_client(self):
        """设置MCP客户端并加载工具"""
        if self.mcp_servers:
            # 确保mcp_servers是一个字典
            if isinstance(self.mcp_servers, dict):
                try:
                    self.mcp_client = MultiServerMCPClient(self.mcp_servers)
                    await self.mcp_client.__aenter__()
                    # 加载MCP工具并添加到工具列表
                    # 检查load_mcp_tools返回的是否为协程对象

                    # 兼容性处理：MultiServerMCPClient可能有get_tools而非list_tools方法
                    mcp_tools_result = None
                    if hasattr(self.mcp_client, "get_tools"):
                        # 使用get_tools方法
                        mcp_tools_result = self.mcp_client.get_tools()
                    else:
                        # 使用标准的load_mcp_tools函数
                        mcp_tools_result = load_mcp_tools(self.mcp_client)

                    if asyncio.iscoroutine(mcp_tools_result):
                        mcp_tools = await mcp_tools_result
                    else:
                        mcp_tools = mcp_tools_result

                    # 确保返回的是可迭代对象
                    if hasattr(mcp_tools, "__iter__"):
                        self.tools.extend(mcp_tools)
                    else:
                        raise TypeError(
                            f"MCP工具加载函数应该返回可迭代对象，当前类型为: {type(mcp_tools)}"
                        )
                except Exception as e:
                    # 确保在出错时关闭MCP客户端
                    if self.mcp_client:
                        try:
                            await self.mcp_client.__aexit__(type(e), e, None)
                        except Exception:
                            pass
                    self.mcp_client = None
                    raise
            else:
                raise TypeError(
                    f"mcp_servers应该是字典类型，当前类型为: {type(self.mcp_servers)}"
                )

    async def _cleanup_mcp_client(self):
        """清理MCP客户端资源"""
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
            self.mcp_client = None

    async def __aenter__(self):
        """异步上下文管理支持"""
        await self._setup_mcp_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文退出时清理资源"""
        await self._cleanup_mcp_client()
