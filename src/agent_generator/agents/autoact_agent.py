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
        )
        
        self.reflect_system_prompt = (
            "你是反思代理(Reflect-Agent)，负责分析信息并得出结论。"
            "审查所有收集到的信息，评估有效性和相关性，并提供一个全面准确的最终答案。"
            "你的回答应该直接解决原始问题，并提供清晰的解释。"
        )

    def _create_agent(self) -> CompiledGraph:
        """
        创建AutoAct代理的工作流图
        
        Returns:
            CompiledGraph: 编译后的代理图
        """
        # 创建状态图
        workflow = StateGraph(AutoActState)
        
        # 添加三个核心节点: 规划、工具使用、反思
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
        elif state.get("final_answer"):
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
        """规划节点，负责分解任务和制定计划"""
        messages = state.get("messages", [])
        
        # 获取工具描述
        tools_desc = self._format_tools_description()
        
        # 构造规划提示
        system_message = SystemMessage(content=self.plan_system_prompt)
        
        # 获取用户问题
        user_query = ""
        for msg in messages:
            if msg.type == "human":
                user_query = msg.content
                break
        
        # 构造规划提示
        planning_prompt = f"""
        请为解决以下问题制定详细的执行计划：

        问题: {user_query}

        可用工具:
        {tools_desc}

        请将问题分解为具体步骤，每个步骤应包括:
        1. 步骤描述: 这一步要做什么
        2. 使用工具: 应该使用哪个工具
        3. 工具参数: 需要提供哪些参数
        
        输出格式应为JSON数组，每个步骤是一个对象，例如:
        [
            {{"description": "步骤1描述", "tool": "工具名称", "parameters": {{"参数1": "值1", "参数2": "值2"}}}},
            {{"description": "步骤2描述", "tool": "工具名称", "parameters": {{"参数1": "值1"}}}}
        ]
        """
        
        # 创建规划消息
        planning_messages = [system_message, HumanMessage(content=planning_prompt)]
        
        # 调用模型生成计划
        plan_response = await self.model.ainvoke(planning_messages)
        plan_content = plan_response.content
        
        # 提取JSON计划
        try:
            # 尝试从响应中提取JSON
            json_match = re.search(r'\[.*\]', plan_content, re.DOTALL)
            if json_match:
                plan_json = json.loads(json_match.group(0))
            else:
                # 如果没有找到JSON，尝试解析整个响应
                plan_json = json.loads(plan_content)
                
            # 确保计划是列表形式
            if not isinstance(plan_json, list):
                raise ValueError("计划必须是列表形式")
                
            # 更新状态
            logger.info(f"生成计划: {plan_json}")
            scratchpad = state.get("scratchpad", "")
            scratchpad += f"\n规划结果:\n{json.dumps(plan_json, ensure_ascii=False, indent=2)}"
            
            return {
                **state,
                "plan": plan_json,
                "messages": state["messages"] + [AIMessage(content=plan_content)],
                "scratchpad": scratchpad
            }
            
        except Exception as e:
            logger.error(f"解析计划失败: {e}")
            # 创建一个简单的默认计划
            default_plan = [{"description": "直接回答问题", "tool": "final_answer", "parameters": {}}]
            scratchpad = state.get("scratchpad", "")
            scratchpad += f"\n解析计划失败，使用默认计划: {default_plan}"
            
            return {
                **state,
                "plan": default_plan,
                "messages": state["messages"] + [AIMessage(content=f"计划生成失败: {e}，使用默认计划")],
                "scratchpad": scratchpad
            }
    
    async def _tool_node(self, state: AutoActState):
        """工具节点，负责调用工具执行计划步骤"""
        messages = state.get("messages", [])
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        
        # 检查是否还有步骤需要执行
        if current_step >= len(plan):
            logger.info("没有更多步骤需要执行")
            return {
                **state,
                "current_step": current_step  # 保持当前步骤不变
            }
        
        # 获取当前步骤
        step = plan[current_step]
        tool_name = step.get("tool", "")
        
        # 如果是final_answer，直接跳到反思阶段
        if tool_name == "final_answer":
            logger.info("当前步骤是final_answer，跳到反思阶段")
            return {
                **state,
                "current_step": len(plan),  # 设置为计划长度，表示计划已完成
                "current_agent": SubAgentType.REFLECT
            }
        
        # 构造工具调用提示
        system_message = SystemMessage(content=self.tool_system_prompt)
        
        # 获取工具描述
        tool = self._find_tool_by_name(tool_name)
        tool_desc = f"工具名称: {tool_name}\n描述: {getattr(tool, 'description', '无描述')}"
        if hasattr(tool, "args_schema"):
            schema_props = getattr(tool.args_schema, "__annotations__", {})
            tool_desc += f"\n参数: {schema_props}"
        
        # 构造工具调用提示
        tool_prompt = f"""
        请为以下计划步骤构造精确的工具调用参数:
        
        步骤说明: {step.get('description', '无描述')}
        
        工具信息:
        {tool_desc}
        
        预期参数:
        {step.get('parameters', {})}
        
        请构造一个有效的JSON对象，包含工具所需的所有参数。只输出参数JSON，不要包含其他解释。
        """
        
        # 创建工具调用消息
        tool_messages = [system_message, HumanMessage(content=tool_prompt)]
        
        # 调用模型生成工具参数
        tool_response = await self.model.ainvoke(tool_messages)
        tool_content = tool_response.content
        
        # 提取JSON参数
        try:
            # 尝试从响应中提取JSON
            json_match = re.search(r'\{.*\}', tool_content, re.DOTALL)
            if json_match:
                tool_params = json.loads(json_match.group(0))
            else:
                # 如果没有找到JSON，尝试解析整个响应
                tool_params = json.loads(tool_content)
            
            # 确保参数是字典形式
            if not isinstance(tool_params, dict):
                raise ValueError("工具参数必须是字典形式")
            
            # 实际调用工具
            logger.info(f"调用工具: {tool_name}，参数: {tool_params}")
            
            tool_result = None
            if tool:
                # 如果工具是异步的
                if asyncio.iscoroutinefunction(getattr(tool, "_run", None)) or asyncio.iscoroutinefunction(getattr(tool, "run", None)):
                    run_method = getattr(tool, "_run", None) or getattr(tool, "run", None)
                    tool_result = await run_method(**tool_params)
                else:
                    # 同步工具
                    run_method = getattr(tool, "_run", None) or getattr(tool, "run", None)
                    tool_result = run_method(**tool_params)
            else:
                tool_result = f"工具 {tool_name} 不存在或无法调用"
            
            # 更新状态
            tool_calls = state.get("tool_calls", [])
            tool_calls.append({
                "tool": tool_name,
                "parameters": tool_params,
                "result": str(tool_result)
            })
            
            observations = state.get("observations", [])
            observations.append(f"工具 {tool_name} 返回结果: {tool_result}")
            
            scratchpad = state.get("scratchpad", "")
            scratchpad += f"\n步骤 {current_step + 1}: 使用工具 {tool_name}\n参数: {json.dumps(tool_params, ensure_ascii=False)}\n结果: {tool_result}\n"
            
            # 更新消息历史
            updated_messages = state["messages"] + [
                AIMessage(content=f"执行步骤 {current_step + 1}: {step.get('description', '')}")
            ]
            
            return {
                **state,
                "current_step": current_step + 1,  # 增加步骤计数
                "tool_calls": tool_calls,
                "observations": observations,
                "scratchpad": scratchpad,
                "messages": updated_messages
            }
            
        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            # 记录失败
            observations = state.get("observations", [])
            observations.append(f"工具 {tool_name} 调用失败: {str(e)}")
            
            scratchpad = state.get("scratchpad", "")
            scratchpad += f"\n步骤 {current_step + 1}: 使用工具 {tool_name} 失败 - {str(e)}\n"
            
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
        
        # 获取用户问题
        user_query = ""
        for msg in messages:
            if msg.type == "human":
                user_query = msg.content
                break
        
        # 整合所有观察结果
        all_observations = "\n".join(observations)
        
        # 构造反思提示
        reflect_prompt = f"""
        请基于以下信息回答原始问题:
        
        原始问题: {user_query}
        
        执行步骤与观察结果:
        {all_observations}
        
        请提供全面、准确的最终答案。直接回答问题，无需解释你的思考过程。
        """
        
        # 创建反思消息
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
        
        return {
            **state,
            "final_answer": final_answer,
            "is_complete": True,
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