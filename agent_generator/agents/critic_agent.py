# -*- coding: utf-8 -*-
"""
@File    :   critic_agent.py
@Time    :   2025/04/23 14:33:25
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   使用CRITIC框架实现的具有自我批评和修正能力的代理
"""

from typing import List, Dict, Any, Optional, Annotated, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool
from .base import BaseAgent
from .templates.prompts import (
    DEFAULT_CRITIC_SYSTEM_PROMPT,
    DEFAULT_CRITIQUE_PROMPT,
    DEFAULT_CORRECTION_PROMPT
)
from langgraph.graph.graph import CompiledGraph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class CriticState(TypedDict):
    """简化的CRITIC代理状态定义"""
    messages: Annotated[list, add_messages]  # 消息历史
    current_answer: Union[str, list]  # 当前答案
    critiques: Union[str, list]  # 当前批评意见
    iterations: int  # 当前迭代次数
    is_complete: bool  # 是否完成
    original_input: Union[str, Dict[str, Any], List[Dict[str, Any]]]  # 原始用户输入
    step: str  # 当前步骤
    system_prompt: str  # 系统提示，用于各节点构建提示


class CriticAgent(BaseAgent):
    """
    实现CRITIC框架的代理，能够通过工具交互进行自我批评和修正 (改进版图结构)
    
    CRITIC框架流程:
    1. 初始化: 生成初始答案
    2. 验证: 使用外部工具(可选)验证答案，生成批评意见
    3. 停止条件: 如果批评意见表明答案正确，返回答案
    4. 修正: 根据批评意见，使用外部工具(可选)修正答案
    5. 迭代: 重复验证和修正过程，直到满足停止条件
    """

    def __init__(
        self,
        name: str,
        model,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
        max_iterations: int = 3,
        use_markers: bool = False,  # 已废弃参数，保留以兼容旧代码
        # 向后兼容的mcp_config参数
        mcp_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化CRITIC代理

        Args:
            name: 代理名称
            model: 使用的语言模型
            system_prompt: 系统提示，定义代理行为
            temperature: 温度参数，控制输出随机性
            tools: 预定义的工具列表
            mcp_servers: 多MCP服务器配置字典
            max_iterations: 最大迭代次数
            use_markers: 是否使用标记（已废弃，保留以兼容旧代码）
            mcp_config: 单个MCP配置字典（向后兼容）
        """
        # 处理兼容性
        if mcp_config and not mcp_servers:
            if not isinstance(mcp_config, dict):
                raise TypeError(f"mcp_config应该是字典类型，当前类型为: {type(mcp_config)}")
            mcp_config_copy = mcp_config.copy()
            if "transport" not in mcp_config_copy:
                mcp_config_copy["transport"] = "stdio"
            mcp_servers = {f"{name}_mcp": mcp_config_copy}

        super().__init__(name, model, tools=tools, mcp_servers=mcp_servers)
        self.system_prompt = system_prompt or DEFAULT_CRITIC_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_iterations = max_iterations
        # 忽略use_markers参数


    def _create_agent(self) -> CompiledGraph:
        """
        创建一个基于CRITIC框架的代理，实现验证-修正循环 (改进版图结构)

        Returns:
            CompiledGraph: 编译后的代理图
        """
        # 创建状态图
        graph_builder = StateGraph(CriticState)
        
        # 获取工具列表
        tools = self.tools or []
        
        # 绑定工具到语言模型 (如果模型支持)
        if hasattr(self.model, "bind_tools"):
             llm_with_tools = self.model.bind_tools(tools)
        else:
             # 对于不支持bind_tools的模型，工具调用可能需要在外部处理
             # 或者依赖于模型的特定工具调用格式
             # 这里简化处理，假设模型能理解工具并在响应中指示调用
             llm_with_tools = self.model
             print("Warning: Model does not support bind_tools. Tool calling might rely on prompt engineering.")


        # --- 节点定义 ---

        def initialize(state: CriticState):
            """根据用户输入生成初始回答"""
            # 提取用户输入
            messages = state["messages"]
            user_message = next((msg for msg in messages if msg.type == "human"), None)
            original_input = user_message.content if user_message else None
            
            # 添加系统提示（如果不存在）
            system_msg = next((msg for msg in messages if msg.type == "system"), None)
            system_prompt = system_msg.content if system_msg else self.system_prompt
            
            if not system_msg:
                messages = [SystemMessage(content=system_prompt)] + messages
            
            # 确保不会重复添加系统消息
            cleaned_messages = []
            has_system = False
            for msg in messages:
                if msg.type == "system":
                    if not has_system:  # 只添加第一个系统消息
                        cleaned_messages.append(msg)
                        has_system = True
                else:
                    cleaned_messages.append(msg)
            
            # 检查历史消息中是否有工具调用结果
            # 每当执行初始化节点时，先清理历史中的验证和修正请求消息
            if len(cleaned_messages) > 2:  # 有系统消息、用户消息，再加上可能的其他消息
                for i in range(len(cleaned_messages) - 1, 1, -1):  # 从后向前，跳过系统和第一个用户消息
                    msg = cleaned_messages[i]
                    # 保留工具结果消息，但移除验证和修正请求和回复
                    if msg.type == "human" and any(keyword in str(msg.content) for keyword in 
                                                ["验证", "批评", "分析以下输出", "根据以下批评"]):
                        # 找到验证或修正请求，移除它和它后面的AI回复
                        cleaned_messages.pop(i)
                        if i < len(cleaned_messages) and cleaned_messages[i].type == "ai":
                            cleaned_messages.pop(i)
            
            # 调用LLM生成初始答案
            response = llm_with_tools.invoke(cleaned_messages)
            
            # 添加到消息历史
            cleaned_messages.append(response)
            
            # 返回状态
            return {
                "messages": cleaned_messages,  # 更新后的消息历史
                "current_answer": response.content,  # 存储当前答案
                "critiques": "",
                "iterations": 0,
                "is_complete": False,
                "original_input": original_input,
                "step": "initialize",
                "system_prompt": system_prompt  # 存储系统提示
            }

        def verify(state: CriticState):
            """验证当前答案"""
            # 获取当前状态
            messages = state["messages"]
            current_answer = state["current_answer"]
            original_input = state["original_input"]
            system_prompt = state["system_prompt"]
            
            # 构建验证提示
            # 获取最近的工具调用消息，如果有的话
            tool_results = [msg for msg in messages if msg.type == "tool"][-3:] if messages else []
            tool_context = ""
            if tool_results:
                tool_outputs = [f"工具调用结果: {tool.content}" for tool in tool_results]
                tool_context = "根据以下工具调用结果:\n" + "\n".join(tool_outputs) + "\n\n"
            
            # 格式化验证提示
            verify_prompt = tool_context + DEFAULT_CRITIQUE_PROMPT.format(
                input=original_input if isinstance(original_input, str) else "多模态输入...",
                current_output=current_answer if isinstance(current_answer, str) else "多模态回答..."
            )
            
            # 创建验证请求消息
            verify_request = HumanMessage(content=verify_prompt)
            
            # 为验证请求创建新的消息列表，避免历史消息过长
            verify_messages = [
                SystemMessage(content=system_prompt),
                verify_request
            ]
            
            # 调用LLM进行验证
            verification_response = llm_with_tools.invoke(verify_messages)
            
            # 不修改主消息列表，只返回批评结果
            return {
                "critiques": verification_response.content,
                "step": "verify"
            }

        def correct(state: CriticState):
            """修正答案"""
            # 获取当前状态
            messages = state["messages"]
            current_answer = state["current_answer"]
            critiques = state["critiques"]
            original_input = state["original_input"]
            iterations = state["iterations"]
            system_prompt = state["system_prompt"]
            
            # 构建修正提示
            # 获取最近的工具调用消息，如果有的话
            tool_results = [msg for msg in messages if msg.type == "tool"][-3:] if messages else []
            tool_context = ""
            if tool_results:
                tool_outputs = [f"工具调用结果: {tool.content}" for tool in tool_results]
                tool_context = "根据以下工具调用结果:\n" + "\n".join(tool_outputs) + "\n\n"
            
            # 格式化修正提示
            correction_prompt = tool_context + DEFAULT_CORRECTION_PROMPT.format(
                input=original_input if isinstance(original_input, str) else "多模态输入...",
                current_output=current_answer if isinstance(current_answer, str) else "多模态回答...",
                critiques=critiques if isinstance(critiques, str) else "多模态批评..."
            )
            
            # 创建修正请求消息
            correction_request = HumanMessage(content=correction_prompt)
            
            # 为修正请求创建新的消息列表，避免历史消息过长
            correction_messages = [
                SystemMessage(content=system_prompt),
                correction_request
            ]
            
            # 调用LLM进行修正
            correction_response = llm_with_tools.invoke(correction_messages)
            
            # 更新用户可见消息历史，替换最后一条AI消息
            updated_messages = messages.copy()
            # 找到最后一条非工具AI消息进行替换
            ai_indices = [i for i, msg in enumerate(updated_messages) if msg.type == "ai" and not isinstance(msg, ToolMessage)]
            if ai_indices:
                last_ai_index = ai_indices[-1]
                updated_messages[last_ai_index] = correction_response
            else:
                # 如果没有找到AI消息，就追加
                updated_messages.append(correction_response)
            
            # 返回更新状态
            return {
                "messages": updated_messages,  # 更新消息历史
                "current_answer": correction_response.content,  # 更新当前答案
                "iterations": iterations + 1,
                "step": "correct"
            }

        # --- 图构建 ---
        
        # 添加节点
        graph_builder.add_node("initialize", initialize)
        graph_builder.add_node("verify", verify)
        graph_builder.add_node("correct", correct)
        
        # 创建工具节点(使用默认的messages_key)
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)
        
        # 设置入口点
        graph_builder.set_entry_point("initialize")

        # 添加边和条件
        
        # 初始化/修正后检查工具调用
        graph_builder.add_conditional_edges(
            "initialize",
            tools_condition,
            {
                "tools": "tools",
                "__end__": "verify"
            }
        )
        
        # 验证节点的条件分支
        graph_builder.add_conditional_edges(
            "verify",
            self._check_verify_output,
            {
                "tools": "tools",    
                "correct": "correct",
                "complete": END        
            }
        )
        
        # 修正节点的条件分支
        graph_builder.add_conditional_edges(
            "correct",
            tools_condition,
            {
                "tools": "tools",
                "__end__": "verify"
            }
        )
        
        # 工具节点返回后的处理
        graph_builder.add_conditional_edges(
            "tools",
            lambda x: x["step"],
            {
                "initialize": "initialize",
                "verify": "verify",
                "correct": "correct"
            }
        )

        # 编译图
        return graph_builder.compile()

    # --- Helper methods for conditional edges ---
    def _check_verify_output(self, state: CriticState):
        """检查验证输出，决定调用工具、修正或完成"""
        # 首先检查是否需要调用工具
        if tools_condition(state) == "tools":
             # 更新步骤标记
             state["step"] = "verify" # 标记是verify触发了工具调用
             # print("Debug: Verify output requires tools.")
             return "tools"
        else:
             # 不需要工具，直接决定是修正还是完成
             decision = self._should_continue_logic(state) # 调用决策逻辑
             # print(f"Debug: No tools needed after verify. Decision: {decision}")
             return decision # 返回"correct"或"complete"



    def _should_continue_logic(self, state: CriticState) -> str:
        """决定是否继续迭代"""
        critiques = state["critiques"]
        iterations = state["iterations"]
        
        # 达到最大迭代次数
        if iterations >= self.max_iterations:
            state["is_complete"] = True
            return "complete"
        
        # 批评为空或明确表示正确
        if isinstance(critiques, str):
            critique_text = critiques.strip()
            positive_indicators = ["未发现错误", "答案是正确的", "验证通过", "没有问题"]
            
            if not critique_text or any(indicator in critique_text for indicator in positive_indicators):
                state["is_complete"] = True
                return "complete"
            else:
                return "correct"
        else:
            # 多模态批评
            return "correct"  # 默认需要修正

    # --- Standard Agent Methods ---
    def _get_messages(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[BaseMessage]:
        """获取消息列表，支持文本和多模态输入"""
        return super()._get_messages(input_data)

    async def agenerate(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]], image_path: Optional[str] = None, image_url: Optional[str] = None, messages: Optional[List[BaseMessage]] = None, **kwargs):
        """异步生成回复，支持文本和图像输入"""
        return await super().agenerate(input_data, image_path, image_url, messages, **kwargs)
