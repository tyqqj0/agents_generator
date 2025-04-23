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
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 步骤标记
STEP_MARKERS = {
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
    messages: Annotated[list, add_messages]  # 消息历史
    current_answer: str  # 当前答案
    critiques: str  # 当前批评意见
    iterations: int  # 当前迭代次数
    is_complete: bool  # 是否完成
    original_input: Union[str, Dict[str, Any], List[Dict[str, Any]]]  # 原始用户输入，支持多模态
    step: str  # 当前步骤


class CriticAgent(BaseAgent):
    """
    实现CRITIC框架的代理，能够通过工具交互进行自我批评和修正
    
    CRITIC框架流程:
    1. 初始化: 生成初始答案
    2. 验证: 使用外部工具验证答案，生成批评意见
    3. 停止条件: 如果批评意见表明答案正确，返回答案
    4. 修正: 基于批评意见修正答案
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
        # 处理兼容性：如果提供了旧的mcp_config，转换为新格式的mcp_servers
        if mcp_config and not mcp_servers:
            if not isinstance(mcp_config, dict):
                raise TypeError(
                    f"mcp_config应该是字典类型，当前类型为: {type(mcp_config)}"
                )

            mcp_config_copy = mcp_config.copy()
            if "transport" not in mcp_config_copy:
                mcp_config_copy["transport"] = "stdio"
            mcp_servers = {f"{name}_mcp": mcp_config_copy}

        super().__init__(name, model, tools=tools, mcp_servers=mcp_servers)
        self.system_prompt = system_prompt or DEFAULT_CRITIC_SYSTEM_PROMPT
        
        # 如果启用了标记，添加标记说明到系统提示
        if use_markers:
            markers_explanation = """
请注意每个步骤的标记:
- [输入] 表示用户的问题
- [初始回答] 表示你的初始回答
- [验证请求] 表示对回答进行验证的请求
- [批评意见] 表示对回答的批评和建议
- [修正请求] 表示基于批评修正回答的请求
- [修正回答] 表示根据批评修正后的回答
"""
            self.system_prompt += markers_explanation
        
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.use_markers = use_markers
        
        # 根据use_markers开关决定是否使用标记
        if not use_markers:
            # 如果不使用标记，将所有标记设为空字符串
            global STEP_MARKERS
            STEP_MARKERS = {k: "" for k in STEP_MARKERS}

    def _create_agent(self) -> CompiledGraph:
        """
        创建一个基于CRITIC框架的代理，实现验证-修正循环

        Returns:
            CompiledGraph: 编译后的代理图
        """
        # 创建状态图
        graph_builder = StateGraph(CriticState)
        
        # 获取工具列表
        tools = self.tools or []
        
        # 绑定工具到语言模型
        llm_with_tools = self.model.bind_tools(tools)

        # 初始化节点：生成初始答案
        def initialize(state: CriticState):
            """根据用户输入生成初始回答"""
            messages = state["messages"]
            
            # 提取用户原始输入
            original_input = None
            for message in messages:
                if message.type == "human":
                    original_input = message.content
                    break
            
            # 添加系统消息
            if messages and messages[0].type != "system":
                messages = [
                    SystemMessage(content=self.system_prompt)
                ] + messages
            
            # 为用户输入添加标记
            if self.use_markers:
                for i, message in enumerate(messages):
                    if message.type == "human" and isinstance(message.content, str):
                        if not message.content.startswith(STEP_MARKERS["input"]):
                            messages[i] = HumanMessage(content=f"{STEP_MARKERS['input']} {message.content}")
            
            # 调用模型生成初始回答
            response = llm_with_tools.invoke(messages)
            
            # 提取文本内容作为当前答案，并添加标记
            if isinstance(response.content, str):
                response_content = response.content
                current_answer = f"{STEP_MARKERS['initial_answer']} {response_content}"
                
                # 创建带标记的回答消息
                marked_response = AIMessage(content=current_answer)
            else:
                # 处理多模态回答
                current_answer = response.content
                marked_response = response
            
            # 返回更新的状态
            return {
                "messages": messages + [marked_response],  # 添加回答到消息历史
                "current_answer": current_answer,  # 保存当前答案
                "critiques": "",  # 初始无批评
                "iterations": 0,  # 初始迭代次数
                "is_complete": False,  # 未完成
                "original_input": original_input,  # 保存原始输入
                "step": "initialize"  # 当前步骤
            }

        # 验证节点：使用工具验证当前答案
        def verify(state: CriticState):
            """使用工具验证当前答案"""
            current_answer = state["current_answer"]
            iterations = state["iterations"]
            messages = state["messages"].copy()
            original_input = state["original_input"]
            
            # 准备用于模板的参数
            if isinstance(original_input, str):
                input_display = original_input
            else:
                # 如果是多模态输入，尝试提取文本部分或使用描述性文本
                input_display = "多模态输入（包含图像或其他非文本内容）"
            
            if isinstance(current_answer, str):
                current_output_display = current_answer
                # 如果有标记，移除以便显示纯内容
                for marker in STEP_MARKERS.values():
                    if marker and current_output_display.startswith(marker):
                        current_output_display = current_output_display[len(marker):].strip()
            else:
                # 多模态答案
                current_output_display = "多模态回答（包含非文本内容）"
            
            # 使用模板构建验证提示
            verify_prompt_template = DEFAULT_CRITIQUE_PROMPT
            verify_prompt = verify_prompt_template.format(
                input=input_display,
                current_output=current_output_display
            )
            
            # 添加标记
            if self.use_markers:
                verify_prompt = f"{STEP_MARKERS['verify_request']} {verify_prompt}"
            
            # 创建验证请求
            verify_messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=verify_prompt)
            ]
            
            # 调用模型进行验证
            verification_response = llm_with_tools.invoke(verify_messages)
            
            # 提取批评意见
            if isinstance(verification_response.content, str):
                verification_content = verification_response.content
                
                # 确保批评意见带有标记
                if self.use_markers and not verification_content.startswith(STEP_MARKERS["critique"]):
                    critiques = f"{STEP_MARKERS['critique']} {verification_content}"
                else:
                    critiques = verification_content
                
                # 创建带标记的批评消息
                critique_message = AIMessage(content=critiques)
            else:
                # 处理多模态批评
                critiques = verification_response.content
                critique_message = verification_response
            
            # 更新消息历史
            messages.append(HumanMessage(content=verify_prompt))
            messages.append(critique_message)
            
            # 返回更新的状态
            return {
                "messages": messages,
                "critiques": critiques,
                "iterations": iterations,
                "step": "verify"
            }

        # 判断是否需要继续修正
        def should_continue(state: CriticState):
            """判断是否需要继续迭代修正"""
            critiques = state["critiques"]
            iterations = state["iterations"]
            
            # 如果到达最大迭代次数，停止
            if iterations >= self.max_iterations:
                return "complete"
                
            # 分析批评意见，判断是否需要继续修正
            # 如果批评是字符串，才进行文本分析
            if isinstance(critiques, str):
                # 检查批评是否表明答案已经正确
                positive_indicators = [
                    "未发现错误",
                    "答案是正确的",
                    "验证通过",
                    "没有发现问题",
                    "答案准确无误"
                ]
                
                # 如果批评中包含表明答案正确的指标，完成迭代
                if any(indicator in critiques for indicator in positive_indicators):
                    return "complete"
            
            # 默认继续修正
            return "correct"
        
        # 修正节点：根据批评修正答案
        def correct(state: CriticState):
            """根据批评意见修正当前答案"""
            current_answer = state["current_answer"]
            critiques = state["critiques"]
            iterations = state["iterations"]
            messages = state["messages"].copy()
            original_input = state["original_input"]
            
            # 准备用于模板的参数
            if isinstance(original_input, str):
                input_display = original_input
            else:
                # 如果是多模态输入，尝试提取文本部分或使用描述性文本
                input_display = "多模态输入（包含图像或其他非文本内容）"
            
            if isinstance(current_answer, str):
                current_output_display = current_answer
                # 如果有标记，移除以便显示纯内容
                for marker in STEP_MARKERS.values():
                    if marker and current_output_display.startswith(marker):
                        current_output_display = current_output_display[len(marker):].strip()
            else:
                # 多模态答案
                current_output_display = "多模态回答（包含非文本内容）"
            
            if isinstance(critiques, str):
                critiques_display = critiques
                # 移除批评意见中的标记
                if self.use_markers and critiques_display.startswith(STEP_MARKERS["critique"]):
                    critiques_display = critiques_display[len(STEP_MARKERS["critique"]):].strip()
            else:
                # 多模态批评
                critiques_display = "多模态批评（包含非文本内容）"
            
            # 使用模板构建修正提示
            correction_prompt_template = DEFAULT_CORRECTION_PROMPT
            correction_prompt = correction_prompt_template.format(
                input=input_display,
                current_output=current_output_display,
                critiques=critiques_display
            )
            
            # 添加标记
            if self.use_markers:
                correction_prompt = f"{STEP_MARKERS['correction_request']} {correction_prompt}"
            
            # 创建修正请求
            correction_messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=correction_prompt)
            ]
            
            # 调用模型进行修正
            correction_response = llm_with_tools.invoke(correction_messages)
            
            # 提取修正后的答案
            if isinstance(correction_response.content, str):
                correction_content = correction_response.content
                
                # 确保修正答案带有标记
                if self.use_markers and not correction_content.startswith(STEP_MARKERS["corrected_answer"]):
                    corrected_answer = f"{STEP_MARKERS['corrected_answer']} {correction_content}"
                else:
                    corrected_answer = correction_content
                
                # 创建带标记的修正消息
                correction_message = AIMessage(content=corrected_answer)
            else:
                # 处理多模态修正答案
                corrected_answer = correction_response.content
                correction_message = correction_response
            
            # 更新消息历史
            messages.append(HumanMessage(content=correction_prompt))
            messages.append(correction_message)
            
            # 返回更新的状态
            return {
                "messages": messages,
                "current_answer": corrected_answer,
                "iterations": iterations + 1,
                "step": "correct"
            }
            
        # 完成节点：标记流程完成并准备最终答案
        def complete(state: CriticState):
            """标记流程完成并准备最终答案"""
            current_answer = state["current_answer"]
            messages = state["messages"].copy()
            iterations = state["iterations"]
            
            # 如果是字符串答案，处理标记
            if isinstance(current_answer, str):
                # 提取不带标记的最终答案
                final_answer = current_answer
                for marker in STEP_MARKERS.values():
                    if marker and final_answer.startswith(marker):
                        final_answer = final_answer[len(marker):].strip()
                
                # 添加最终答案信息到消息历史
                completion_message = f"{STEP_MARKERS['complete']} 经过{iterations}次迭代验证和修正后，最终答案如下:\n\n{final_answer}"
                final_message = AIMessage(content=completion_message)
                messages.append(final_message)
            else:
                # 多模态答案直接使用
                final_message = AIMessage(content=f"经过{iterations}次迭代验证和修正后，最终答案如下")
                messages.append(final_message)
                # 可以添加多模态答案本身，但这取决于实现细节
            
            return {
                "messages": messages,
                "is_complete": True,
                "step": "complete"
            }
            
        # 添加节点
        graph_builder.add_node("initialize", initialize)
        graph_builder.add_node("verify", verify)
        graph_builder.add_node("correct", correct)
        graph_builder.add_node("complete", complete)
        
        # 添加边和条件边
        graph_builder.add_edge("initialize", "verify")
        graph_builder.add_conditional_edges(
            "verify", 
            should_continue,
            {
                "correct": "correct",
                "complete": "complete"
            }
        )
        graph_builder.add_edge("correct", "verify")
        
        # 设置入口点
        graph_builder.set_entry_point("initialize")
        
        # 编译并返回图
        return graph_builder.compile()

    def _get_messages(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[BaseMessage]:
        """
        获取消息列表，支持文本和多模态输入
        
        Args:
            input_data: 输入数据
                - 字符串: 作为纯文本消息处理
                - 字典: 包含多模态内容的消息
                - 列表: 多个内容块组成的消息
        
        Returns:
            List[BaseMessage]: 消息列表
        """
        # 直接使用BaseAgent的_get_messages实现，确保多模态兼容性
        return super()._get_messages(input_data)

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
        # 直接使用BaseAgent的agenerate实现，确保多模态兼容性
        return await super().agenerate(input_data, image_path, image_url, messages, **kwargs)
