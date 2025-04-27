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

import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f"Initializing CriticAgent '{name}'...")
        # 处理兼容性
        if mcp_config and not mcp_servers:
            logger.info("Using legacy mcp_config parameter.")
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
        logger.info(f"CriticAgent '{name}' initialized with max_iterations={max_iterations}.")


    def _create_agent(self) -> CompiledGraph:
        """
        创建一个基于CRITIC框架的代理，实现验证-修正循环 (改进版图结构)

        Returns:
            CompiledGraph: 编译后的代理图
        """
        logger.info("Creating CriticAgent graph...")
        # 创建状态图
        graph_builder = StateGraph(CriticState)
        
        # 获取工具列表
        tools = self.tools or []
        logger.info(f"Agent tools: {[tool.name for tool in tools]}")
        
        # 绑定工具到语言模型 (如果模型支持)
        bound_model = self.model # Create a separate variable for the bound model
        if hasattr(self.model, "bind_tools"):
             logger.info("Binding tools to model.")
             bound_model = self.model.bind_tools(tools)
        else:
             logger.warning("Model does not support bind_tools. Tool calling might rely on prompt engineering.")

        # --- 节点定义 ---

        def initialize(state: CriticState):
            logger.info("Node: initialize", color="green")
            """根据用户输入生成初始回答"""
            messages = state["messages"]
            
            # 检查是否是从工具节点返回
            is_from_tool = any(msg.type == "tool" for msg in messages)
            
            if not is_from_tool:
                # 首次执行initialize节点
                user_message = next((msg for msg in messages if msg.type == "human"), None)
                original_input = user_message.content if user_message else None
                logger.info(f"Original input type: {type(original_input)}")
                
                system_msg = next((msg for msg in messages if msg.type == "system"), None)
                system_prompt = system_msg.content if system_msg else self.system_prompt
                
                initial_messages = []
                if not system_msg:
                    logger.info("Adding system prompt.")
                    initial_messages.append(SystemMessage(content=system_prompt))
                else:
                    initial_messages.append(system_msg)
                
                if user_message:
                    initial_messages.append(user_message)
            else:
                # 从工具节点返回，需要处理工具结果
                tool_messages = [msg for msg in messages if msg.type == "tool"]
                user_message = next((msg for msg in messages if msg.type == "human"), None)
                original_input = user_message.content if user_message else None
                system_prompt = state.get("system_prompt", self.system_prompt)
                
                # 构建包含工具结果的消息
                initial_messages = [SystemMessage(content=system_prompt)]
                if user_message:
                    initial_messages.append(user_message)
                # 添加最近的工具结果
                initial_messages.extend(tool_messages[-3:])
            
            # 调用模型生成回答
            logger.info(f"Invoking model for initial answer with {len(initial_messages)} messages.")
            response = bound_model.invoke(initial_messages)
            
            # 检查响应是否为空
            if not response.content or (isinstance(response.content, str) and not response.content.strip()):
                logger.warning("Empty response received, creating default content")
                response.content = "无法生成回答，请尝试重新提问或提供更多信息。"
            
            logger.info(f"Initial response received. Content length: {len(response.content) if isinstance(response.content, str) else 'Non-string content'}")
            
            # 用logger比info更高级的
            logger.debug(f"Initial response preview: {str(response.content)[:100]}...")
            # 同时，还应该修改图结构，确保tools节点执行后不再返回initialize节点
            return {
                "messages": [user_message, response] if user_message else [response],
                "current_answer": response.content,
                "critiques": "",
                "iterations": 0,
                "is_complete": False,
                "original_input": original_input,
                "step": "verify",  # 修改为verify，确保下一步是验证而不是再次初始化
                "system_prompt": system_prompt
            }

        def verify(state: CriticState):
            logger.info("Node: verify", color="green")
            """验证当前答案"""
            messages = state["messages"].copy()
            current_answer = state["current_answer"]
            original_input = state["original_input"]
            system_prompt = state["system_prompt"]
            
            tool_results = [msg for msg in messages if msg.type == "tool"][-3:]
            tool_context = ""
            if tool_results:
                logger.info(f"Found {len(tool_results)} recent tool results for verification context.")
                tool_outputs = [f"Tool Call Result: {tool.content}" for tool in tool_results]
                tool_context = "Based on the following tool results:\n" + "\n".join(tool_outputs) + "\n\n"
            
            verify_prompt = tool_context + DEFAULT_CRITIQUE_PROMPT.format(
                input=original_input if isinstance(original_input, str) else "[Multimodal Input]",
                current_output=current_answer if isinstance(current_answer, str) else "[Multimodal Answer]"
            )
            verify_prompt += "\n\nYou can use available tools to verify the information's accuracy. Please use tools proactively when needed."
            verify_request = HumanMessage(content=verify_prompt)
            
            # Use a minimal context for the verification call itself
            verify_call_messages = [SystemMessage(content=system_prompt), verify_request]
            logger.info(f"Invoking model for verification with {len(verify_call_messages)} messages.")
            verification_response = bound_model.invoke(verify_call_messages)
            
            critique_content = verification_response.content
            if not critique_content or (isinstance(critique_content, str) and critique_content.strip() == ""):
                logger.warning("Received empty verification response, treating as need correction")
                critique_content = "Verification result was empty, assuming correction is needed."
            
            logger.info(f"Verification response received. Critique length: {len(critique_content) if isinstance(critique_content, str) else 'Non-string critique'}")
            logger.debug(f"Verification response preview: {str(critique_content)[:100]}...")
            
            # Add the request and response to the main message history for tool condition check
            updated_messages = messages + [verify_request, verification_response]
            
            return {
                "messages": updated_messages,
                "critiques": critique_content,
                "step": "verify"
            }

        def correct(state: CriticState):
            logger.info("Node: correct", color="green")
            """修正答案"""
            messages = state["messages"].copy()
            current_answer = state["current_answer"]
            critiques = state["critiques"]
            original_input = state["original_input"]
            iterations = state["iterations"]
            system_prompt = state["system_prompt"]
            
            tool_results = [msg for msg in messages if msg.type == "tool"][-3:]
            tool_context = ""
            if tool_results:
                logger.info(f"Found {len(tool_results)} recent tool results for correction context.")
                tool_outputs = [f"Tool Call Result: {tool.content}" for tool in tool_results]
                tool_context = "Based on the following tool results:\n" + "\n".join(tool_outputs) + "\n\n"
            
            correction_prompt = tool_context + DEFAULT_CORRECTION_PROMPT.format(
                input=original_input if isinstance(original_input, str) else "[Multimodal Input]",
                current_output=current_answer if isinstance(current_answer, str) else "[Multimodal Answer]",
                critiques=critiques if isinstance(critiques, str) else "[Multimodal Critique]"
            )
            correction_prompt += "\n\nYou can use available tools to find more accurate information."
            correction_request = HumanMessage(content=correction_prompt)
            
            # Use minimal context for correction call
            correct_call_messages = [SystemMessage(content=system_prompt), correction_request]
            logger.info(f"Invoking model for correction with {len(correct_call_messages)} messages.")
            correction_response = bound_model.invoke(correct_call_messages)
            
            logger.info(f"Correction response received. Content length: {len(correction_response.content) if isinstance(correction_response.content, str) else 'Non-string content'}")
            logger.debug(f"Correction response preview: {str(correction_response.content)[:100]}...")
            
            # Update main message history: Add request and response
            updated_messages = messages + [correction_request, correction_response]
            new_answer = correction_response.content
            
            return {
                "messages": updated_messages,
                "current_answer": new_answer,
                "iterations": iterations + 1,
                "step": "correct"
            }

        # --- 图构建 ---
        logger.info("Building graph nodes and edges...")
        graph_builder = StateGraph(CriticState)
        
        graph_builder.add_node("initialize", initialize)
        graph_builder.add_node("verify", verify)
        graph_builder.add_node("correct", correct)
        
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)
        
        graph_builder.set_entry_point("initialize")
        logger.info("Entry point set to 'initialize'.")

        # --- Edges --- 
        graph_builder.add_conditional_edges(
            "initialize",
            tools_condition, # Check if the response from 'initialize' requires tools
            {
                "tools": "tools",
                "__end__": "verify"
            }
        )
        logger.info("Added conditional edges from 'initialize'.")
        
        graph_builder.add_conditional_edges(
            "verify",
            self._check_verify_output, # Custom logic to check tools OR decide continue/complete
            {
                "tools": "tools",    
                "correct": "correct",
                "complete": END        
            }
        )
        logger.info("Added conditional edges from 'verify'.")
        
        graph_builder.add_conditional_edges(
            "correct",
            tools_condition, # Check if the response from 'correct' requires tools
            {
                "tools": "tools",
                "__end__": "verify"
            }
        )
        logger.info("Added conditional edges from 'correct'.")
        
        # After tools are called, route back based on the step that triggered the tools
        graph_builder.add_conditional_edges(
            "tools",
            lambda x: x.get("step", "initialize"),
            {
                "initialize": "initialize", # Should ideally not happen, but fallback
                "verify": "verify",         # If verify node called tools, go back to verify
                "correct": "correct"        # If correct node called tools, go back to correct
            }
        )

        logger.info("Added conditional edges from 'tools' back to the triggering node.")
                
        # graph_builder.add_conditional_edges(
        #     "tools",
        #     lambda x: "verify",  # 直接返回verify，不再根据step返回
        #     {
        #         "verify": "verify"  # 工具执行后总是进入verify节点
        #     }
        # )

        compiled_graph = graph_builder.compile()
        logger.info("Graph compilation complete.")
        return compiled_graph

    # --- Helper methods for conditional edges ---
    def _check_verify_output(self, state: CriticState):
        logger.debug("Checking verification output...")
        # Check if the LAST message (which should be the verification_response) has tool calls
        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        needs_tools = isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls
        
        if needs_tools:
             logger.info("Tools needed after verify node based on AIMessage tool_calls.")
             state["step"] = "verify"
             return "tools"
        else:
             decision = self._should_continue_logic(state)
             logger.info(f"No tools needed after verify. Decision based on critique: {decision}")
             return decision

    def _should_continue_logic(self, state: CriticState) -> str:
        critiques = state["critiques"]
        iterations = state["iterations"]
        logger.info(f"Checking continue logic: Iterations={iterations}, Max Iterations={self.max_iterations}")
        logger.debug(f"Critique content preview: {str(critiques)[:100]}...")
        
        if iterations >= self.max_iterations:
            logger.info(f"Reached max iterations ({self.max_iterations}), completing.")
            state["is_complete"] = True
            return "complete"
        
        if isinstance(critiques, str):
            critique_text = critiques.strip()
            
            if not critique_text:
                logger.warning("Empty critique text received, continuing correction.")
                return "correct"
            
            # Adjusted positive indicators for stricter checking
            positive_indicators = ["答案是正确的", "验证通过", "无需修正", "准确无误"]
            # Check if ANY positive indicator is present
            is_positive = any(indicator in critique_text for indicator in positive_indicators)
            # Also check for very short answers that might imply correctness
            is_short_positive = len(critique_text) < 15 and ("好" in critique_text or "对" in critique_text or "ok" in critique_text.lower())

            if is_positive or is_short_positive:
                logger.info(f"Positive or short positive critique received: '{critique_text[:50]}...', completing.")
                state["is_complete"] = True
                return "complete"
            else:
                logger.info("Critique is not clearly positive, continuing correction.")
                return "correct"
        else:
            logger.info("Critique is not a string (multimodal?), continuing correction.")
            return "correct"

    # --- Standard Agent Methods ---
    def _get_messages(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[BaseMessage]:
        return super()._get_messages(input_data)

    async def agenerate(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]], image_path: Optional[str] = None, image_url: Optional[str] = None, messages: Optional[List[BaseMessage]] = None, **kwargs):
        logger.info(f"agenerate called for agent '{self.name}'.")
        return await super().agenerate(input_data, image_path, image_url, messages, **kwargs)