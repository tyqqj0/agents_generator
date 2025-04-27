# -*- coding: utf-8 -*-
"""
@File    :   react_agent.py
@Time    :   2025/04/23 14:33:25
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""






from typing import List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from .base import BaseAgent
from .templates.prompts import DEFAULT_TOOL_PROMPT
from langgraph.graph.graph import CompiledGraph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt import create_react_agent


class ReactAgent(BaseAgent):
    """基于React的智能代理"""

    def __init__(
        self,
        name: str,
        model,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
        # 向后兼容的mcp_config参数
        mcp_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化React代理

        参数:
            name: 代理名称
            model: 使用的语言模型
            system_prompt: 系统提示，定义代理行为
            temperature: 温度参数，控制输出随机性
            tools: 预定义的工具列表
            mcp_servers: 多MCP服务器配置字典，新推荐方式
            mcp_config: 单个MCP配置字典，向后兼容旧接口
        """
        super().__init__(name, model, tools=tools, mcp_servers=mcp_servers)
        self.system_prompt = system_prompt or DEFAULT_TOOL_PROMPT
        self.temperature = temperature

    def _create_agent(self) -> CompiledGraph:
        """
        创建一个基于LangGraph的React代理

        Returns:
            CompiledGraph: 编译后的LangGraph图
        """
        # 使用LangGraph的create_react_agent函数创建代理
        return create_react_agent(
            model=self.model,
            tools=self.tools,
        )
        
    def _get_messages(self, input_text: str) -> List[BaseMessage]:
        """
        获取代理的初始消息列表

        Args:
            input_text: 用户输入的初始文本

        Returns:
            List[BaseMessage]: 初始消息列表
        """
        return [HumanMessage(content=input_text)]
    
