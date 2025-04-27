

# -*- coding: utf-8 -*-
"""
@File    :   null_agent.py
@Time    :   2025/04/24 15:12:30
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""


from typing import List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from .base import BaseAgent
from .templates.prompts import DEFAULT_PROMPT
from langgraph.graph.graph import CompiledGraph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class NullAgent(BaseAgent):
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
            mcp_servers: 此处无用
            mcp_config: 此处无用
        """


        super().__init__(name, model)
        self.system_prompt = system_prompt or DEFAULT_PROMPT
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
            response = self.model.invoke(messages)

            # 返回包含响应的新状态
            return {"messages": [response]}

        # 添加聊天机器人节点
        graph_builder.add_node("chatbot", chatbot)

        # 设置入口点
        graph_builder.set_entry_point("chatbot")

        # 编译并返回图
        return graph_builder.compile()
    
    

    def _get_messages(self, input_text: str) -> List[BaseMessage]:
        """提供带有系统提示的消息列表"""
        from langchain_core.messages import SystemMessage
        
        # 创建包含系统提示的消息列表
        messages = [SystemMessage(content=self.system_prompt)]
        # 添加用户输入
        messages.append(HumanMessage(content=input_text))
        
        return messages