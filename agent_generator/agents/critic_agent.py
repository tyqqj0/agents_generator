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

# 步骤标记 (根据 use_markers 开关动态调整)
STEP_MARKERS_DEFAULT = {
    "input": "[输入]",
    "initial_answer": "[初始回答]",
    "verify_request": "[验证请求]",
    "critique": "[批评意见]",
    "correction_request": "[修正请求]",
    "corrected_answer": "[修正回答]",
    "complete": "[完成]"
}

class CriticState(TypedDict):
    """CRITIC代理的状态定义"""
    messages: Annotated[list, add_messages]  # 完整内部消息历史
    current_answer: Union[str, list] # 当前答案（可能是多模态）
    critiques: Union[str, list] # 当前批评意见（可能是多模态）
    iterations: int  # 当前迭代次数
    is_complete: bool  # 是否完成
    original_input: Union[str, Dict[str, Any], List[Dict[str, Any]]]  # 原始用户输入，支持多模态
    step: str  # 当前步骤
    user_facing_messages: List[BaseMessage]  # 用户可见的消息历史


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
        use_markers: bool = True,  # 是否使用标记
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
            use_markers: 是否使用标记（如[输入]、[初始回答]等）
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
        
        # 处理标记
        self.use_markers = use_markers
        self.step_markers = STEP_MARKERS_DEFAULT.copy() # 每个实例有自己的标记副本
        if use_markers:
            markers_explanation = "\n请注意每个步骤的标记:\n" + "\n".join([f"- {v} 表示{k}" for k, v in self.step_markers.items()])
            self.system_prompt += markers_explanation
        else:
            self.step_markers = {k: "" for k in self.step_markers}

        self.temperature = temperature
        self.max_iterations = max_iterations


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
            messages = state["messages"]
            
            # 提取原始输入
            original_input = next((msg.content for msg in messages if msg.type == "human"), None)

            # 确保有系统消息
            if not messages or messages[0].type != "system":
                messages = [SystemMessage(content=self.system_prompt)] + (messages or [])

            # 标记用户输入
            marked_messages = messages[:1] # Keep system prompt
            for msg in messages[1:]:
                if msg.type == "human" and isinstance(msg.content, str):
                     content = msg.content
                     if self.use_markers and not content.startswith(self.step_markers["input"]):
                          content = f"{self.step_markers['input']} {content}"
                     marked_messages.append(HumanMessage(content=content))
                else:
                     marked_messages.append(msg) # Keep other types as is

            user_facing_messages = marked_messages.copy()

            # 调用LLM生成初始回答
            response = llm_with_tools.invoke(marked_messages) # Use marked messages for invocation
            
            # 处理回答并标记
            current_answer = response.content
            if isinstance(current_answer, str):
                if self.use_markers and not current_answer.startswith(self.step_markers["initial_answer"]):
                    current_answer = f"{self.step_markers['initial_answer']} {current_answer}"
                marked_response = AIMessage(content=current_answer)
            else:
                # 多模态回答不加标记
                marked_response = response

            # 更新用户可见消息
            user_facing_messages.append(marked_response)
            
            return {
                "messages": marked_messages + [marked_response], # Store marked messages and response internally
                "current_answer": response.content, # Store raw content
                "critiques": "",
                "iterations": 0,
                "is_complete": False,
                "original_input": original_input,
                "step": "initialize",
                "user_facing_messages": user_facing_messages
            }

        def verify(state: CriticState):
            """使用工具(可选)验证当前答案，生成批评意见"""
            current_answer_raw = state["current_answer"] # Use raw answer
            iterations = state["iterations"]
            messages = state["messages"].copy() # Full internal history
            original_input = state["original_input"]
            user_facing_messages = state["user_facing_messages"].copy()

            # --- 处理工具调用返回的结果 ---
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                 # 如果上一步是工具调用，将其结果整合到验证提示中
                 tool_output = last_message.content
                 # (这里可以根据需要，将工具结果更智能地融入提示)
                 critique_context = f"基于以下工具调用结果:\n{tool_output}\n\n请继续验证。"
                 print(f"Debug: Integrating tool result into verification: {tool_output}")
            else:
                 critique_context = ""

            # --- 准备验证提示 ---
            input_display = "多模态输入..." if not isinstance(original_input, str) else original_input
            current_output_display = "多模态回答..." if not isinstance(current_answer_raw, str) else current_answer_raw
            
            verify_prompt_template = DEFAULT_CRITIQUE_PROMPT
            verify_prompt_content = verify_prompt_template.format(
                input=input_display,
                current_output=current_output_display
            )
            # 添加工具结果上下文
            verify_prompt_content += f"\n{critique_context}"

            # 添加标记
            if self.use_markers:
                verify_prompt_content = f"{self.step_markers['verify_request']} {verify_prompt_content}"
            
            # 创建验证请求 (只用于LLM调用，不加入user_facing)
            verify_request_message = HumanMessage(content=verify_prompt_content)
            
            # 调用模型进行验证
            # 使用完整的内部消息历史 + 新的验证请求
            verification_response = llm_with_tools.invoke(messages + [verify_request_message])
            
            # --- 处理验证结果 ---
            critiques_raw = verification_response.content
            if isinstance(critiques_raw, str):
                if self.use_markers and not critiques_raw.startswith(self.step_markers["critique"]):
                     critiques_marked = f"{self.step_markers['critique']} {critiques_raw}"
                else:
                     critiques_marked = critiques_raw
                critique_message_for_history = AIMessage(content=critiques_marked) # Marked for internal history
                critique_message_for_user = AIMessage(content=critiques_raw) # Raw for potential user display later? (Decide later)
            else:
                # 多模态批评
                critiques_marked = critiques_raw
                critique_message_for_history = verification_response
                critique_message_for_user = verification_response # Raw

            return {
                # Add verify request and response to internal history
                "messages": messages + [verify_request_message, critique_message_for_history],
                "critiques": critiques_raw, # Store raw critique content
                "step": "verify",
                # user_facing_messages 不变，直到修正或完成
                "user_facing_messages": user_facing_messages
            }

        def correct(state: CriticState):
            """根据批评意见，使用工具(可选)修正答案"""
            current_answer_raw = state["current_answer"]
            critiques_raw = state["critiques"]
            iterations = state["iterations"]
            messages = state["messages"].copy()
            original_input = state["original_input"]
            user_facing_messages = state["user_facing_messages"].copy()

            # --- 处理工具调用返回的结果 ---
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                 tool_output = last_message.content
                 correction_context = f"基于以下工具调用结果进行修正:\n{tool_output}\n\n"
                 print(f"Debug: Integrating tool result into correction: {tool_output}")
            else:
                 correction_context = ""

            # --- 准备修正提示 ---
            input_display = "多模态输入..." if not isinstance(original_input, str) else original_input
            current_output_display = "多模态回答..." if not isinstance(current_answer_raw, str) else current_answer_raw
            critiques_display = "多模态批评..." if not isinstance(critiques_raw, str) else critiques_raw

            correction_prompt_template = DEFAULT_CORRECTION_PROMPT
            correction_prompt_content = correction_prompt_template.format(
                input=input_display,
                current_output=current_output_display,
                critiques=critiques_display
            )
            # 添加工具结果上下文
            correction_prompt_content = correction_context + correction_prompt_content

            # 添加标记
            if self.use_markers:
                correction_prompt_content = f"{self.step_markers['correction_request']} {correction_prompt_content}"

            # 创建修正请求 (只用于LLM调用，不加入user_facing)
            correction_request_message = HumanMessage(content=correction_prompt_content)

            # 调用模型进行修正
            correction_response = llm_with_tools.invoke(messages + [correction_request_message])

            # --- 处理修正结果 ---
            corrected_answer_raw = correction_response.content
            if isinstance(corrected_answer_raw, str):
                 if self.use_markers and not corrected_answer_raw.startswith(self.step_markers["corrected_answer"]):
                      corrected_answer_marked = f"{self.step_markers['corrected_answer']} {corrected_answer_raw}"
                 else:
                      corrected_answer_marked = corrected_answer_raw
                 correction_message_for_history = AIMessage(content=corrected_answer_marked)
                 correction_message_for_user = AIMessage(content=corrected_answer_raw) # Raw for user
            else:
                 # 多模态修正
                 corrected_answer_marked = corrected_answer_raw
                 correction_message_for_history = correction_response
                 correction_message_for_user = correction_response # Raw

            # 更新用户可见消息 (替换上一个AI消息)
            if user_facing_messages and user_facing_messages[-1].type == "ai":
                 user_facing_messages[-1] = correction_message_for_user
            else:
                 user_facing_messages.append(correction_message_for_user)

            return {
                # Add correction request and response to internal history
                "messages": messages + [correction_request_message, correction_message_for_history],
                "current_answer": corrected_answer_raw, # Store raw corrected answer
                "iterations": iterations + 1,
                "step": "correct",
                "user_facing_messages": user_facing_messages # Updated user messages
            }

        # --- 图构建 ---

        # 添加节点
        graph_builder.add_node("initialize", initialize)
        graph_builder.add_node("verify", verify)
        graph_builder.add_node("correct", correct)
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)
        
        # 注意：'complete' 节点现在由 END 代替

        # 设置入口点
        graph_builder.set_entry_point("initialize")

        # 添加边
        
        # 初始化后总是去验证
        graph_builder.add_edge("initialize", "verify")

        # Conditional edge after Verify: Decide tools, correct, or complete
        graph_builder.add_conditional_edges(
            "verify",
            # This custom function checks for tool calls first, then decides correct/complete
            self._check_verify_output,
            {
                "tools": "tools",      # Call tools if needed
                "correct": "correct",  # Go to correct node if decision is "correct"
                "complete": END         # End graph if decision is "complete"
            }
        )
        
        # Conditional edge after Correct: Decide tools or go back to verify
        graph_builder.add_conditional_edges(
            "correct",
            # This custom function checks for tool calls first, then decides the next step
            self._check_correct_output,
            {
                "tools": "tools",      # Call tools if needed
                "continue": "verify"   # Otherwise, go back to verify the corrected answer
            }
        )

        # Edges from tools node - route back to the appropriate node
        graph_builder.add_conditional_edges(
            "tools",
            lambda state: state["step"], # Route based on the step *before* tool call
            {
                "verify": "verify",   # If verify called tools, go back to verify
                "correct": "correct", # If correct called tools, go back to correct
                # Add other potential callers if needed (e.g., initialize if needed)
            }
        )

        # 编译图
        return graph_builder.compile()

    # --- Helper methods for conditional edges ---
    def _check_verify_output(self, state: CriticState):
        """Checks verify output: call tools or decide correct/complete"""
        # First, check if the last message from 'verify' node requests tool usage
        if tools_condition(state) == "tools":
             # Update step marker *before* going to tools
             state["step"] = "verify" # Mark that 'verify' triggered the tool call
             print("Debug: Verify output requires tools.")
             return "tools"
        else:
             # No tools needed, directly decide whether to correct or complete
             # We need to call the logic originally in the should_continue node here
             decision = self._should_continue_logic(state) # Call the decision logic
             print(f"Debug: No tools needed after verify. Decision: {decision}")
             return decision # Return "correct" or "complete"

    def _check_correct_output(self, state: CriticState):
        """Checks correct output: call tools or continue to verify"""
        # First, check if the last message from 'correct' node requests tool usage
        if tools_condition(state) == "tools":
             # Update step marker *before* going to tools
             state["step"] = "correct" # Mark that 'correct' triggered the tool call
             print("Debug: Correct output requires tools.")
             return "tools"
        else:
             print("Debug: No tools needed after correct. Proceeding to verify.")
             return "continue" # No tools needed, go back to verify

    def _should_continue_logic(self, state: CriticState) -> str:
        """Internal logic to decide whether to continue iteration. Returns 'correct' or 'complete'."""
        critiques = state["critiques"]
        iterations = state["iterations"]

        if iterations >= self.max_iterations:
            print(f"Debug: Max iterations ({self.max_iterations}) reached.")
            return "complete"

        if isinstance(critiques, str):
             critique_text = critiques
             # Remove marker for analysis if necessary (assuming it's already raw in state)
             # if self.use_markers and critique_text.startswith(self.step_markers['critique']):
             #      critique_text = critique_text[len(self.step_markers['critique']):].strip()

             positive_indicators = ["未发现错误", "答案是正确的", "验证通过", "没有发现问题", "答案准确无误"]
             if not critique_text or critique_text.strip() == "" or any(indicator in critique_text for indicator in positive_indicators):
                  print(f"Debug: Stopping criteria met. Critique: '{critique_text[:100]}...'")
                  return "complete"
             else:
                  print(f"Debug: Correction needed based on critique: '{critique_text[:100]}...'")
                  return "correct"
        else:
             # Assume multimodal critiques always need correction (or add specific logic)
             print("Debug: Multimodal critique received, proceeding to correction.")
             return "correct"

    # --- Standard Agent Methods ---
    def _get_messages(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[BaseMessage]:
        """获取消息列表，支持文本和多模态输入"""
        return super()._get_messages(input_data)

    async def agenerate(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]], image_path: Optional[str] = None, image_url: Optional[str] = None, messages: Optional[List[BaseMessage]] = None, **kwargs):
        """异步生成回复，支持文本和图像输入"""
        return await super().agenerate(input_data, image_path, image_url, messages, **kwargs)
