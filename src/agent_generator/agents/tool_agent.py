# -*- coding: utf-8 -*-
"""
@File    :   tool_agent.py
@Time    :   2025/04/23 14:33:30
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""






from typing import List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool
from .base import BaseAgent
from .templates.prompts import DEFAULT_TOOL_PROMPT
from langgraph.graph.graph import CompiledGraph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class ToolAgent(BaseAgent):
    """能够使用工具的代理"""

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
        初始化工具代理

        Args:
            name: 代理名称
            model: 使用的语言模型
            system_prompt: 系统提示，定义代理行为
            temperature: 温度参数，控制输出随机性
            tools: 预定义的工具列表
            mcp_servers: 多MCP服务器配置字典，新推荐方式
                格式: {"server_name": {"command": "...", "args": [...], "transport": "..."}}
            mcp_config: 单个MCP配置字典，向后兼容旧接口
                如果提供，会转换为一个包含单个服务器的mcp_servers字典
        """
        # 处理兼容性：如果提供了旧的mcp_config，转换为新格式的mcp_servers
        if mcp_config and not mcp_servers:
            # 确保mcp_config是字典类型
            if not isinstance(mcp_config, dict):
                raise TypeError(
                    f"mcp_config应该是字典类型，当前类型为: {type(mcp_config)}"
                )

            # 默认transport为stdio
            mcp_config_copy = mcp_config.copy()  # 创建副本避免修改原始对象
            if "transport" not in mcp_config_copy:
                mcp_config_copy["transport"] = "stdio"
            mcp_servers = {f"{name}_mcp": mcp_config_copy}

        super().__init__(name, model, tools=tools, mcp_servers=mcp_servers)
        self.system_prompt = system_prompt or DEFAULT_TOOL_PROMPT
        self.temperature = temperature

    def _create_agent(self) -> CompiledGraph:
        """
        创建一个基于LangGraph的支持工具调用的对话代理

        Returns:
            CompiledGraph: 编译后的代理图
        """

        # 定义状态类型
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        # 创建状态图构建器
        graph_builder = StateGraph(State)

        # 获取工具列表，确保包含MCP工具
        tools = self.tools or []
        
        # 在绑定工具前添加日志，验证工具是否正确加载
        # print(f"tools: {tools}")
        
        # 绑定工具到语言模型
        llm_with_tools = self.model.bind_tools(tools)

        # 定义聊天机器人节点函数
        def chatbot(state: State):
            # 使用系统提示创建完整的消息列表
            messages = state["messages"]
            if messages and messages[0].type != "system":
                # 如果没有系统消息，添加系统提示
                messages = [
                    {"type": "system", "content": self.system_prompt}
                ] + messages
                
                
            # 检查模型

            # 调用语言模型
            response = llm_with_tools.invoke(messages)

            # 返回包含响应的新状态
            return {"messages": [response]}

        # 添加聊天机器人节点
        graph_builder.add_node("chatbot", chatbot)

        # 创建工具节点
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)

        # 添加条件边缘，根据模型输出决定是否调用工具
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )

        # 工具执行完毕后返回到聊天机器人节点
        graph_builder.add_edge("tools", "chatbot")

        # 设置入口点
        graph_builder.set_entry_point("chatbot")

        # 编译并返回图
        return graph_builder.compile()
    
    

    def _get_messages(self, input_text: str) -> List[BaseMessage]:
        """提供带有系统提示的消息列表"""
        # 创建包含系统提示的消息列表
        messages = []
        
        # 只有当系统提示不为空时才添加系统消息
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        # 添加用户输入，支持字符串和其他格式
        if isinstance(input_text, str):
            messages.append(HumanMessage(content=input_text))
        else:
            # 处理多模态内容
            messages.append(HumanMessage(content=input_text))
        
        return messages