# -*- coding: utf-8 -*-
"""
@File    :   autoact_agent.py
@Time    :   2025/04/23 18:45:25
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   基于AutoAct框架实现的自动学习代理
"""

from typing import List, Dict, Any, Optional, Union, Set, Callable, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from .base import BaseAgent
from pydantic import Field
import json
import re
import logging
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.graph import CompiledGraph
from typing_extensions import TypedDict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_plan_from_content(plan_content: str) -> List[Dict[str, Any]]:
    """
    从计划内容中提取结构化的步骤列表
    
    Args:
        plan_content: 模型生成的计划文本
        
    Returns:
        List[Dict[str, Any]]: 结构化的计划步骤列表
    """
    # 尝试从计划内容中提取JSON格式的步骤
    json_pattern = r'\[.*\]'
    json_match = re.search(json_pattern, plan_content, re.DOTALL)
    if json_match:
        try:
            steps = json.loads(json_match.group(0))
            if isinstance(steps, list):
                # 确保每个步骤都有必要的字段
                for i, step in enumerate(steps):
                    if not isinstance(step, dict):
                        steps[i] = {"description": str(step), "tool": ""}
                    elif "description" not in step:
                        steps[i]["description"] = f"步骤 {i+1}"
                    if "tool" not in step:
                        steps[i]["tool"] = ""
                return steps
        except (json.JSONDecodeError, ValueError):
            pass
    
    # 尝试从格式化文本中提取步骤
    steps = []
    step_pattern = r'(?:步骤|Step)\s*(\d+)[:.：]?\s*(.*?)(?=(?:步骤|Step)\s*\d+[:.：]?|$)'
    step_matches = re.finditer(step_pattern, plan_content, re.DOTALL | re.IGNORECASE)
    
    for match in step_matches:
        step_num = match.group(1)
        description = match.group(2).strip()
        
        # 尝试提取工具名称
        tool_match = re.search(r'(?:使用|Use)\s+([A-Za-z0-9_]+)(?:\s+工具)?', description, re.IGNORECASE)
        tool_name = tool_match.group(1) if tool_match else ""
        
        steps.append({
            "description": description,
            "tool": tool_name
        })
    
    # 如果上述方法都未提取出步骤，则尝试按行分割
    if not steps:
        lines = [line.strip() for line in plan_content.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            # 跳过可能的计划标题
            if i == 0 and ("计划" in line or "Plan" in line):
                continue
                
            # 尝试提取工具名称
            tool_match = re.search(r'(?:使用|Use)\s+([A-Za-z0-9_]+)(?:\s+工具)?', line, re.IGNORECASE)
            tool_name = tool_match.group(1) if tool_match else ""
            
            steps.append({
                "description": line,
                "tool": tool_name
            })
    
    # 如果仍然没有提取出步骤，则将整个内容作为一个步骤
    if not steps:
        steps = [{"description": plan_content.strip(), "tool": ""}]
    
    return steps

class SubAgentType:
    """定义子代理类型"""
    PLAN = "plan"
    TOOL = "tool"
    REFLECT = "reflect"

class AutoActState(TypedDict):
    """AutoAct代理的状态定义"""
    messages: Annotated[list, add_messages]  # 消息历史
    observations: List[str]  # 观察结果
    current_step: int  # 当前步骤
    plan: Optional[List[Dict[str, Any]]]  # 计划步骤
    tool_calls: List[Dict[str, Any]]  # 工具调用历史
    final_answer: Optional[str]  # 最终回答
    is_complete: bool  # 是否完成
    current_agent: str  # 当前活跃的子代理
    scratchpad: str  # 工作记录板

class AutoActAgent(BaseAgent):
    """
    基于AutoAct框架实现的自动学习代理
    
    特点:
    1. 自动工具选择能力
    2. 任务分解与规划能力
    3. 自我分化为专业子代理
    4. 支持群体规划协作
    """
    
    def __init__(
        self,
        name: str,
        model,
        temperature: float = 0.7,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
        training_data: Optional[List[Dict]] = None,  # 少量训练数据
        max_steps: int = 10,  # 最大步骤数
        max_generated_samples: int = 200,  # 自我指导生成的样本数
    ):
        """
        初始化AutoAct代理

        Args:
            name: 代理名称
            model: 使用的语言模型
            temperature: 温度参数
            tools: 预定义的工具列表
            mcp_servers: 多MCP服务器配置
            training_data: 训练数据
            max_steps: 最大执行步骤数
            max_generated_samples: 自我指导生成的最大样本数
        """
        super().__init__(name, model, tools=tools, mcp_servers=mcp_servers)
        self.temperature = temperature
        self.training_data = training_data or []
        self.max_steps = max_steps
        self.max_generated_samples = max_generated_samples
        
        # 子代理系统提示
        self.plan_system_prompt = (
            "你是规划代理(Plan-Agent)，负责任务分解和工具选择。"
            "你需要将复杂问题分解为可执行步骤，并为每个步骤选择合适的工具。"
            "只需输出计划步骤，不需要执行工具调用。"
        )
        
        self.tool_system_prompt = (
            "你是工具代理(Tool-Agent)，负责精确调用工具。"
            "根据提供的计划步骤和工具说明，构造正确的工具调用参数。"
            "你需要准确解析工具需要的参数格式，确保调用成功。"
            "当你认为所有必要的工具调用已完成，或者已获得足够的信息来回答原始问题时，"
            "请明确指出现在应该生成最终答案，例如写：'任务已完成，现在可以生成最终答案'。"
        )
        
        self.reflect_system_prompt = (
            "你是反思代理(Reflect-Agent)，负责分析信息并得出结论。"
            "审查所有收集到的信息，评估有效性和相关性，并提供一个全面准确的最终答案。"
            "你的回答应该直接解决原始问题，并提供清晰的解释。"
            "生成最终答案后，请确保明确标记任务已完成。"
        )
        
        self.execution_system_prompt = (
            "你是执行代理(Execute-Agent)，负责执行单个计划步骤。"
            "根据当前计划步骤的要求，执行相应操作并返回结果。"
            "你需要准确理解步骤需求，并提供清晰的执行结果。"
        )
        
        self.reflection_system_prompt = (
            "你是分析代理(Analysis-Agent)，负责审视执行过程并提供反馈。"
            "分析已完成步骤的结果，评估执行质量，并决定下一步行动。"
            "你需要基于已收集的信息，提供对整体任务进展的反思和建议。"
        )
        
        self.final_system_prompt = (
            "你是总结代理(Summary-Agent)，负责整合所有信息并提供最终回答。"
            "全面审视整个执行过程、所有收集的信息，并提供一个完整、权威的最终回答。"
            "你的回答应直接解决用户的原始问题，清晰、简洁、准确。"
        )

    def _create_agent(self) -> CompiledGraph:
        """
        创建AutoAct代理的工作流图
        
        Returns:
            CompiledGraph: 编译后的代理图
        """
        # 创建状态图
        workflow = StateGraph(AutoActState)
        
        # 添加核心节点: 规划、工具使用、反思
        workflow.add_node("plan_node", self._plan_node)
        workflow.add_node("tool_node", self._tool_node)
        workflow.add_node("reflect_node", self._reflect_node)
        workflow.add_node("route_node", self._route_node)
        
        # 添加路由条件
        def router_condition(state: AutoActState):
            # 判断当前应该路由到哪个节点
            if state.get("is_complete", False):
                return END
            
            current_agent = state.get("current_agent", "")
            
            if current_agent == SubAgentType.PLAN:
                return "plan_node"
            elif current_agent == SubAgentType.TOOL:
                return "tool_node"
            elif current_agent == SubAgentType.REFLECT:
                return "reflect_node"
            else:
                # 默认从路由节点开始
                return "route_node"
                
        # 连接节点和条件
        workflow.add_conditional_edges(
            "route_node",
            router_condition,
            {
                "plan_node": "plan_node",
                "tool_node": "tool_node",
                "reflect_node": "reflect_node",
                "route_node": "route_node",
                END: END
            }
        )
        workflow.add_edge("plan_node", "route_node")
        workflow.add_edge("tool_node", "route_node")
        workflow.add_edge("reflect_node", "route_node")
        
        # 设置入口节点
        workflow.set_entry_point("route_node")
        
        # 编译工作流
        return workflow.compile()
    
    async def _route_node(self, state: AutoActState):
        """路由节点，决定下一步执行哪个子代理"""
        messages = state.get("messages", [])
        
        # 如果是首次执行，检查是否有计划
        if not state.get("plan"):
            # 首次执行，路由到规划代理
            logger.info("首次执行，路由到规划代理")
            return {
                **state,
                "current_agent": SubAgentType.PLAN,
                "current_step": 0,
                "observations": [],
                "tool_calls": [],
                "scratchpad": "",
                "is_complete": False
            }
        
        # 如果已有计划但当前步骤小于计划长度，路由到工具代理
        elif state.get("plan") and state.get("current_step", 0) < len(state.get("plan", [])):
            # 下一步是工具执行
            logger.info(f"有计划步骤待执行，路由到工具代理，当前步骤: {state.get('current_step')}")
            return {
                **state,
                "current_agent": SubAgentType.TOOL
            }
        
        # 如果计划已全部执行完，路由到反思代理
        elif state.get("plan") and state.get("current_step", 0) >= len(state.get("plan", [])):
            # 计划执行完毕，进行反思
            logger.info("计划执行完毕，路由到反思代理")
            return {
                **state,
                "current_agent": SubAgentType.REFLECT
            }
        
        # 如果已有最终答案，标记为完成
        elif state.get("final_answer") or state.get("is_complete"):
            logger.info("已有最终答案，标记为完成")
            return {
                **state,
                "is_complete": True
            }
        
        # 默认路由到规划代理
        logger.info("默认路由到规划代理")
        return {
            **state,
            "current_agent": SubAgentType.PLAN
        }
    
    async def _plan_node(self, state: AutoActState):
        """规划节点，生成行动计划"""
        messages = state.get("messages", [])
        
        system_message = SystemMessage(content=self.plan_system_prompt)
        
        # 检查是否有多模态消息（包含图像）
        has_image = False
        original_multimodal_message = None
        
        for msg in messages:
            if msg.type == "human":
                if isinstance(msg.content, list):
                    # 检测多模态消息中的图像
                    has_image = any(item.get("type") == "image_url" for item in msg.content if isinstance(item, dict))
                    if has_image:
                        original_multimodal_message = msg
                break
        
        # 构建规划消息列表
        if has_image and original_multimodal_message:
            # 如果有图像，使用原始多模态消息
            planning_messages = [system_message, original_multimodal_message]
        else:
            # 否则使用普通消息
            planning_messages = [system_message] + messages
        
        # 调用模型生成计划
        planning_response = await self.model.ainvoke(planning_messages)
        plan = planning_response.content
        
        # 更新状态
        logger.info(f"生成计划: {plan}")
        
        # 提取计划步骤并添加到状态
        plan_steps = extract_plan_from_content(plan)
        if not plan_steps:
            logger.warning("无法从计划中提取步骤，使用整个计划作为一个步骤")
            plan_steps = [{"description": plan, "tool": ""}]
        
        # 初始化记录区
        scratchpad = f"计划:\n{plan}\n\n执行步骤:\n"
        
        return {
            **state,
            "plan": plan,
            "plan_steps": plan_steps,
            "current_step_index": 0,
            "scratchpad": scratchpad,
            "has_image": has_image
        }
    
    async def _tool_node(self, state: AutoActState):
        """工具节点，负责调用工具执行计划步骤"""
        messages = state.get("messages", [])
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        plan_steps = state.get("plan_steps", [])
        scratchpad = state.get("scratchpad", "")
        has_image = state.get("has_image", False)
        
        # 检查是否还有步骤需要执行
        if current_step >= len(plan):
            logger.info("没有更多步骤需要执行")
            return {
                **state,
                "current_step": current_step  # 保持当前步骤不变
            }
        
        # 获取当前步骤
        step = plan[current_step]
        
        # 确保步骤是字典格式
        if not isinstance(step, dict):
            # 如果步骤是字符串或其他非字典类型，转换为字典格式
            step = {"description": str(step), "tool": ""}
            
        tool_name = step.get("tool", "")
        step_description = step.get("description", "无描述")
        
        # 如果是final_answer，直接跳到反思阶段
        if tool_name == "final_answer":
            logger.info("当前步骤是final_answer，跳到反思阶段")
            return {
                **state,
                "current_step": len(plan),  # 设置为计划长度，表示计划已完成
                "current_agent": SubAgentType.REFLECT
            }
        
        # 构建工具调用消息列表
        tool_messages = []
        
        # 添加系统消息
        tool_messages.append(SystemMessage(content=self.tool_system_prompt))
        
        # 查找原始多模态消息并添加
        original_message = None
        image_url = None
        for msg in messages:
            if msg.type == "human":
                if isinstance(msg.content, list):
                    # 处理多模态内容
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            has_image = True
                            if "image_url" in item and "url" in item["image_url"]:
                                image_url = item["image_url"]["url"]
                    original_message = msg
                else:
                    # 普通文本消息
                    original_message = msg
                break
        
        # 如果有图像，添加原始消息让模型可以看到
        if has_image and original_message and isinstance(original_message.content, list):
            tool_messages.append(original_message)
        
        # 创建工具说明提示
        tools_description = ""
        if self.tools:
            for tool in self.tools:
                if getattr(tool, "name", "") == tool_name or not tool_name:
                    tools_description += f"- {getattr(tool, 'name', 'unknown')}: {getattr(tool, 'description', '无描述')}\n"
        
        # 创建工具调用提示
        tool_prompt = f"""
当前需要执行的计划步骤：
{step_description}

可用工具：
{tools_description}

请执行这个步骤，如果需要使用工具，请使用适当的工具来完成任务。

重要提示：
1. 如果你认为所有必要的工具调用已完成，或已获得足够信息回答原始问题
2. 或者这是计划中的最后一个步骤
3. 或者当前步骤已经解决了用户的核心问题

请明确写出"任务已完成，现在可以生成最终答案"，而不是继续调用工具。
"""
        tool_messages.append(HumanMessage(content=tool_prompt))
        
        # 使用绑定工具的模型调用
        llm_with_tools = self.model.bind_tools(self.tools)
        
        # 调用模型
        try:
            logger.info(f"执行步骤 {current_step + 1}: {step_description}")
            response = await llm_with_tools.ainvoke(tool_messages)
            
            # 处理响应
            tool_calls = []
            observations = state.get("observations", [])
            tool_result = None
            
            # 检查响应是否包含完成信号
            completion_indicators = [
                "任务已完成", "可以生成最终答案", "生成最终回答", 
                "回答原始问题", "提供最终结论", "任务完成", 
                "工具调用已完成", "已经获得足够信息"
            ]
            
            is_task_complete = False
            response_content = response.content if hasattr(response, "content") else ""
            
            if response_content:
                # 检查是否有完成信号
                if any(indicator in response_content for indicator in completion_indicators):
                    is_task_complete = True
                    logger.info("检测到任务完成信号，准备生成最终答案")
                    observations.append("检测到任务完成信号，准备生成最终答案")
                    scratchpad += f"\n检测到任务完成信号：{response_content}\n"
            
            # 如果任务已完成，跳转到反思阶段
            if is_task_complete:
                updated_messages = state["messages"] + [AIMessage(content=response_content)]
                return {
                    **state,
                    "messages": updated_messages,
                    "observations": observations,
                    "scratchpad": scratchpad,
                    "current_step": len(plan),  # 设置为计划长度，表示计划已完成
                    "current_agent": SubAgentType.REFLECT
                }
                
            # 检查响应中是否有工具调用
            if hasattr(response, "tool_calls") and response.tool_calls:
                for call in response.tool_calls:
                    tool_name = call.get("name", "")
                    tool_params = call.get("args", {})
                    tool_output = call.get("output", "")
                    
                    # 记录工具调用
                    tool_calls.append({
                        "tool": tool_name,
                        "parameters": tool_params,
                        "result": tool_output
                    })
                    
                    # 添加观察
                    observations.append(f"工具 {tool_name} 返回结果: {tool_output}")
                    
                    # 更新记录区
                    scratchpad += f"\n步骤 {current_step + 1}: 使用工具 {tool_name}\n参数: {json.dumps(tool_params, ensure_ascii=False)}\n结果: {tool_output}\n"
                    
                    # 用于返回的内容
                    tool_result = tool_output
            else:
                # 没有工具调用，直接使用模型输出作为结果
                tool_result = response.content
                observations.append(f"步骤 {current_step + 1} 执行结果: {tool_result}")
                scratchpad += f"\n步骤 {current_step + 1}: {step_description}\n结果: {tool_result}\n"
            
            # 更新消息历史
            updated_messages = state["messages"] + [AIMessage(content=f"执行步骤 {current_step + 1}: {step_description}\n结果: {tool_result}")]
            
            return {
                **state,
                "current_step": current_step + 1,  # 增加步骤计数
                "tool_calls": state.get("tool_calls", []) + tool_calls,
                "observations": observations,
                "scratchpad": scratchpad,
                "messages": updated_messages
            }
            
        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            # 记录失败
            observations = state.get("observations", [])
            observations.append(f"步骤 {current_step + 1} 执行失败: {str(e)}")
            
            scratchpad = state.get("scratchpad", "")
            scratchpad += f"\n步骤 {current_step + 1}: 执行失败 - {str(e)}\n"
            
            # 更新消息历史
            updated_messages = state["messages"] + [
                AIMessage(content=f"执行步骤 {current_step + 1} 失败: {str(e)}")
            ]
            
            return {
                **state,
                "current_step": current_step + 1,  # 即使失败也前进到下一步
                "observations": observations,
                "scratchpad": scratchpad,
                "messages": updated_messages
            }
    
    async def _reflect_node(self, state: AutoActState):
        """反思节点，分析收集的信息并生成最终答案"""
        messages = state.get("messages", [])
        observations = state.get("observations", [])
        
        # 构造反思提示
        system_message = SystemMessage(content=self.reflect_system_prompt)
        
        # 获取用户问题和图像信息
        user_query = ""
        has_image = False
        original_multimodal_message = None
        
        for msg in messages:
            if msg.type == "human":
                if isinstance(msg.content, str):
                    user_query = msg.content
                elif isinstance(msg.content, list):
                    # 处理多模态消息
                    has_image = any(item.get("type") == "image_url" for item in msg.content if isinstance(item, dict))
                    text_parts = [item.get("text", "") for item in msg.content if isinstance(item, dict) and item.get("type") == "text"]
                    user_query = " ".join(text_parts)
                    original_multimodal_message = msg
                break
        
        # 整合所有观察结果
        all_observations = "\n".join(observations)
        
        # 构造反思提示
        reflect_prompt = f"""
        请基于以下信息回答原始问题:
        
        原始问题: {user_query}{"（注意：原始问题包含图像，请在回答中考虑图像内容）" if has_image else ""}
        
        执行步骤与观察结果:
        {all_observations}
        
        请提供全面、准确的最终答案。直接回答问题，清晰地解释你的推理过程。
        注意：生成完最终答案后，请在答案最后添加一行：[任务已完成]
        """
        
        # 创建反思消息
        if has_image and original_multimodal_message:
            # 如果有图像，使用原始多模态消息
            reflect_messages = [system_message, original_multimodal_message, HumanMessage(content=reflect_prompt)]
        else:
            reflect_messages = [system_message, HumanMessage(content=reflect_prompt)]
        
        # 调用模型生成最终答案
        reflect_response = await self.model.ainvoke(reflect_messages)
        final_answer = reflect_response.content
        
        # 更新状态
        logger.info(f"生成最终答案: {final_answer}")
        
        scratchpad = state.get("scratchpad", "")
        scratchpad += f"\n最终回答:\n{final_answer}"
        
        # 更新消息历史
        updated_messages = state["messages"] + [AIMessage(content=final_answer)]
        
        # 检查答案中是否包含完成标记
        completion_markers = ["任务已完成", "Task completed", "[任务已完成]"]
        is_marked_complete = any(marker in final_answer for marker in completion_markers)
        
        logger.info(f"最终答案完成状态: {is_marked_complete}")
        
        return {
            **state,
            "final_answer": final_answer,
            "is_complete": True,  # 始终设置为完成
            "scratchpad": scratchpad,
            "messages": updated_messages
        }
    
    def _format_tools_description(self) -> str:
        """格式化工具描述"""
        descriptions = []
        for tool in self.tools:
            desc = f"- {getattr(tool, 'name', 'unknown')}: {getattr(tool, 'description', '无描述')}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def _find_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """根据名称查找工具"""
        for tool in self.tools:
            if getattr(tool, "name", "") == tool_name:
                return tool
        return None
    
    async def self_instruct(self, seed_samples: List[Dict], sample_count: int = 10):
        """从少量示例自动生成更多训练样本"""
        if not seed_samples:
            logger.warning("没有提供种子样本，无法执行自我指导")
            return []
            
        generated_samples = []
        
        # 构造自我指导提示
        prompt = f"""
        基于以下示例，生成{sample_count}个类似的问题和回答对:
        
        示例:
        {json.dumps(seed_samples, indent=2, ensure_ascii=False)}
        
        请生成{sample_count}个新的、多样化的问题和回答对，格式与示例相同。
        每个生成的问题应该包含多个推理步骤，需要使用工具来解决。
        输出应该是一个有效的JSON数组。
        """
        
        # 调用模型生成新样本
        system_message = SystemMessage(content="你是一个数据生成专家，擅长创建高质量的问答样本数据。")
        messages = [system_message, HumanMessage(content=prompt)]
        
        try:
            response = await self.model.ainvoke(messages)
            response_content = response.content
            
            # 提取JSON
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if json_match:
                new_samples = json.loads(json_match.group(0))
            else:
                new_samples = json.loads(response_content)
                
            # 验证生成的样本
            if isinstance(new_samples, list):
                generated_samples.extend(new_samples)
                logger.info(f"成功生成 {len(new_samples)} 个新样本")
            else:
                logger.error("生成的样本不是列表形式")
        
        except Exception as e:
            logger.error(f"自我指导过程出错: {e}")
        
        return generated_samples

    async def select_tools(self, task_description: str) -> List[Dict[str, Any]]:
        """为特定任务自动选择适合的工具"""
        if not self.tools:
            logger.warning("没有可用工具，无法进行工具选择")
            return []
            
        # 创建工具选择提示
        tools_info = "\n".join([
            f"- {getattr(tool, 'name', 'unknown')}: {getattr(tool, 'description', '无描述')}"
            for tool in self.tools
        ])
        
        prompt = f"""
        任务描述: {task_description}
        
        可用工具列表:
        {tools_info}
        
        请选择对完成上述任务最有用的工具。对于每个选择的工具，给出1-5分的重要性评分和选择理由。
        
        输出格式:
        [
            {{"tool": "工具名称", "score": 5, "reason": "选择理由"}},
            {{"tool": "工具名称", "score": 3, "reason": "选择理由"}}
        ]
        """
        
        # 调用模型进行工具选择
        system_message = SystemMessage(content="你是一个AI助手，擅长为任务选择最合适的工具。")
        messages = [system_message, HumanMessage(content=prompt)]
        
        try:
            response = await self.model.ainvoke(messages)
            response_content = response.content
            
            # 提取JSON
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if json_match:
                selected_tools = json.loads(json_match.group(0))
            else:
                selected_tools = json.loads(response_content)
                
            # 验证工具选择结果
            if isinstance(selected_tools, list):
                # 按重要性评分排序
                selected_tools = sorted(selected_tools, key=lambda x: x.get("score", 0), reverse=True)
                logger.info(f"成功选择 {len(selected_tools)} 个工具")
                return selected_tools
            else:
                logger.error("工具选择结果不是列表形式")
                return []
                
        except Exception as e:
            logger.error(f"工具选择过程出错: {e}")
            return []

    async def _execute_node(self, state: AutoActState):
        """执行节点，负责执行单个计划步骤"""
        messages = state.get("messages", [])
        plan = state.get("plan", "")
        plan_steps = state.get("plan_steps", [])
        current_step_index = state.get("current_step_index", 0)
        scratchpad = state.get("scratchpad", "")
        has_image = state.get("has_image", False)
        
        # 如果已经完成所有步骤，则转到下一个节点
        if current_step_index >= len(plan_steps):
            return {**state, "node": "final"}
        
        # 获取当前步骤
        current_step = plan_steps[current_step_index]
        logger.info(f"执行步骤 {current_step_index + 1}: {current_step}")
        
        # 确保步骤是字典格式
        if not isinstance(current_step, dict):
            # 如果步骤是字符串或其他非字典类型，转换为字典格式
            current_step = {"description": str(current_step), "tool": ""}
            
        step_description = current_step.get("description", "无描述")
        
        # 构建执行消息
        system_message = SystemMessage(content=self.execution_system_prompt)
        
        # 查找原始多模态消息
        original_multimodal_message = None
        if has_image:
            for msg in messages:
                if msg.type == "human" and isinstance(msg.content, list):
                    has_image_url = any(item.get("type") == "image_url" for item in msg.content if isinstance(item, dict))
                    if has_image_url:
                        original_multimodal_message = msg
                        break
        
        # 构建执行消息
        execution_messages = [system_message]
        
        # 添加原始消息
        if has_image and original_multimodal_message:
            # 使用原始多模态消息
            execution_messages.append(original_multimodal_message)
        else:
            # 使用普通消息
            execution_messages.extend(messages)
        
        # 添加计划信息
        plan_info = f"""
我已为解决问题制定了以下计划：
{plan}

当前正在执行步骤 {current_step_index + 1}:
{step_description}

请执行这个步骤，如果需要使用工具，请返回工具调用。如果不需要工具，请直接返回步骤执行结果。
"""
        execution_messages.append(HumanMessage(content=plan_info))
        
        # 调用模型执行步骤
        execution_response = await self.model.ainvoke(execution_messages)
        step_result = execution_response.content
        
        # 更新记录区
        scratchpad += f"\n步骤 {current_step_index + 1}: {step_description}\n结果: {step_result}\n"
        
        # 更新状态，移至下一步骤
        return {
            **state,
            "messages": messages + [AIMessage(content=step_result)],
            "current_step_index": current_step_index + 1,
            "scratchpad": scratchpad
        }

    async def _reflection_node(self, state: AutoActState):
        """反思节点，负责分析执行结果并决定下一步行动"""
        messages = state.get("messages", [])
        plan = state.get("plan", "")
        plan_steps = state.get("plan_steps", [])
        current_step_index = state.get("current_step_index", 0)
        scratchpad = state.get("scratchpad", "")
        has_image = state.get("has_image", False)
        
        # 构建反思消息
        system_message = SystemMessage(content=self.reflection_system_prompt)
        
        # 查找原始多模态消息
        original_multimodal_message = None
        if has_image:
            for msg in messages:
                if msg.type == "human" and isinstance(msg.content, list):
                    has_image_url = any(item.get("type") == "image_url" for item in msg.content if isinstance(item, dict))
                    if has_image_url:
                        original_multimodal_message = msg
                        break
        
        # 构建反思消息
        reflection_messages = [system_message]
        
        # 添加原始消息
        if has_image and original_multimodal_message:
            # 使用原始多模态消息
            reflection_messages.append(original_multimodal_message)
        else:
            # 使用普通消息
            reflection_messages.extend(messages)
        
        # 添加计划和执行历史信息
        reflection_info = f"""
我已为解决问题制定了以下计划：
{plan}

到目前为止已完成以下步骤：
{scratchpad}

请分析执行情况，并决定下一步行动：
1. 如果所有步骤已完成，请总结结果并完成任务
2. 如果需要调整计划，请提供调整后的计划并说明理由
3. 如果执行中遇到问题，请给出解决方案

请提供一个简短的反思分析，以及接下来的行动建议。
"""
        reflection_messages.append(HumanMessage(content=reflection_info))
        
        # 调用模型进行反思
        reflection_response = await self.model.ainvoke(reflection_messages)
        reflection = reflection_response.content
        
        # 更新记录区
        scratchpad += f"\n反思：\n{reflection}\n"
        
        # 检查是否所有步骤都已完成
        if current_step_index >= len(plan_steps):
            # 所有步骤完成，移至最终节点
            return {
                **state,
                "messages": messages + [AIMessage(content=reflection)],
                "scratchpad": scratchpad,
                "node": "final"
            }
        else:
            # 继续执行下一步骤
            return {
                **state,
                "messages": messages + [AIMessage(content=reflection)],
                "scratchpad": scratchpad,
                "node": "execute"
            }

    async def _final_node(self, state: AutoActState):
        """最终节点，整合所有结果并生成最终输出"""
        messages = state.get("messages", [])
        plan = state.get("plan", "")
        scratchpad = state.get("scratchpad", "")
        has_image = state.get("has_image", False)
        
        # 构建最终消息
        system_message = SystemMessage(content=self.final_system_prompt)
        
        # 查找原始多模态消息
        original_multimodal_message = None
        if has_image:
            for msg in messages:
                if msg.type == "human" and isinstance(msg.content, list):
                    has_image_url = any(item.get("type") == "image_url" for item in msg.content if isinstance(item, dict))
                    if has_image_url:
                        original_multimodal_message = msg
                        break
        
        # 构建最终消息列表
        final_messages = [system_message]
        
        # 添加原始消息
        if has_image and original_multimodal_message:
            # 使用原始多模态消息
            final_messages.append(original_multimodal_message)
        else:
            # 使用普通消息
            final_messages.extend(messages[:1])  # 只添加第一条用户消息
        
        # 添加执行历史信息
        final_info = f"""
我已完成了解决问题的所有步骤。

我制定的计划是：
{plan}

执行过程和结果：
{scratchpad}

请基于以上信息，为用户提供一个完整、清晰的最终回答。回答应该：
1. 直接解决用户的原始问题
2. 总结主要发现和结果
3. 如有必要，提供任何相关建议或注意事项

请确保回答简洁明了，直接满足用户需求。
"""
        final_messages.append(HumanMessage(content=final_info))
        
        # 调用模型生成最终回答
        final_response = await self.model.ainvoke(final_messages)
        final_answer = final_response.content
        
        # 更新状态并返回最终结果
        return {
            **state,
            "messages": messages + [AIMessage(content=final_answer)],
            "scratchpad": scratchpad + f"\n最终回答：\n{final_answer}",
            "node": None  # 表示流程结束
        }