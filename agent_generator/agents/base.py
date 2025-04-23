# -*- coding: utf-8 -*-
"""
@File    :   base.py
@Time    :   2025/04/23 14:33:17
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





from typing import List, Dict, Any, Optional, Union, Set
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import asyncio
from contextlib import AsyncExitStack
import base64

# 新增导入MCP工具相关库
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from abc import ABC, abstractmethod
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph




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
        if not isinstance(model, BaseChatModel):
            raise ValueError("model必须是BaseChatModel类型")
        self.model = model
        self.tools = tools or []
        self.mcp_servers = mcp_servers or {}
        self.memory = memory
        self.mcp_client = None
        self.agent = None
    
    # 后初始化，在进入上下文管理器时调用，确保工具绑定
    def _post_init(self):
        self.agent = self._create_agent()
        # print(f"agent: {self.agent}")
    @abstractmethod
    def _create_agent(self) -> CompiledGraph:
        """创建代理, 子类必须实现"""
        pass

    def _get_messages(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[BaseMessage]:
        """
        获取消息列表，支持文本和多模态输入
        
        Args:
            input_data: 输入数据
                - 字符串: 作为纯文本消息处理
                - 字典: 包含多模态内容的消息，可以是图像或其他媒体类型
                - 列表: 多个内容块组成的消息
        
        Returns:
            List[BaseMessage]: 消息列表
        """
        # 如果是字符串，作为纯文本消息处理
        if isinstance(input_data, str):
            return [HumanMessage(content=input_data)]
        
        # 如果是字典或列表，作为多模态消息处理
        elif isinstance(input_data, (dict, list)):
            # 将单个字典转为列表处理
            if isinstance(input_data, dict):
                content = [input_data]
            else:
                content = input_data
                
            return [HumanMessage(content=content)]
        
        # 默认情况，尝试转为字符串处理
        return [HumanMessage(content=str(input_data))]
    
    def _get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """获取适合LLM使用的工具定义，子类可以覆盖以自定义工具构建逻辑"""
        if not self.tools:
            return []
        return [tool.dict() for tool in self.tools if hasattr(tool, "dict")]

    async def agenerate(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]], image_path: Optional[str] = None, image_url: Optional[str] = None, messages: Optional[List[BaseMessage]] = None, **kwargs):
        """
        异步生成回复，支持文本和图像输入
        
        Args:
            input_data: 输入数据
                - 字符串: 纯文本提示
                - 字典: 包含多模态内容的消息数据
                - 列表: 多个内容块组成的消息
            image_path: 可选，本地图像文件路径
            image_url: 可选，图像URL
            messages: 可选，已有的消息历史
            **kwargs: 其他参数
        
        Returns:
            模型响应
        """
        # 如果提供了图像，构建多模态输入
        if image_path or image_url:
            # 构建基本文本内容
            content = [{"type": "text", "text": input_data if isinstance(input_data, str) else str(input_data)}]
            
            # 如果有本地图像路径，读取并编码为base64
            if image_path:
                try:
                    with open(image_path, "rb") as img_file:
                        image_data = base64.b64encode(img_file.read()).decode("utf-8")
                        content.append({
                            "type": "image",
                            "source_type": "base64",
                            "mime_type": "image/jpeg",  # 可根据实际文件类型调整
                            "data": image_data
                        })
                except Exception as e:
                    raise ValueError(f"读取图像文件失败: {str(e)}")
            
            # 如果有图像URL
            elif image_url:
                content.append({
                    "type": "image",
                    "source_type": "url",
                    "url": image_url
                })
            
            # 使用多模态内容
            input_data = content

        # 如果没有绑定工具，则绑定工具
        if not self.tools and self.mcp_servers:
            await self._setup_mcp_client()
            # if_temp = True

            tool_names = [getattr(self.tools[0], "name", str(self.tools[0]))]
            # print(f"绑定工具: {tool_names}")  # 调试信息
        # 准备消息
        message = self._get_messages(input_data)
        # 异步执行代理
        response = await self.agent.ainvoke({"messages": message})

        
        return response


    async def _setup_mcp_client(self):
        """设置MCP客户端并加载工具"""
        # print(f"设置MCP客户端并加载工具: {self.mcp_servers}")
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
        self._post_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文退出时清理资源"""
        await self._cleanup_mcp_client()
