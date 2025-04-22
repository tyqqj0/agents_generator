from typing import Dict, List, Any, Optional, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from .base import BaseAgent, AgentResponse


class RouterAgent(BaseAgent):
    """
    路由代理，根据请求内容将请求路由到合适的专业代理。
    """

    def __init__(
        self,
        name: str,
        model,
        agents: Dict[str, BaseAgent],
        system_prompt: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        初始化路由代理

        Args:
            name: 代理名称
            model: 使用的语言模型（用于路由决策）
            agents: 专业代理字典，格式为 {"agent_name": agent_instance}
            system_prompt: 路由决策的系统提示
            tools: 预定义工具列表
            mcp_servers: MCP服务器配置
        """
        super().__init__(name, model, tools, mcp_servers)
        self.agents = agents

        # 如果未提供系统提示，创建默认的路由提示
        if system_prompt is None:
            agent_descriptions = "\n".join(
                [
                    f"- {name}: {agent.__class__.__name__}"
                    for name, agent in self.agents.items()
                ]
            )
            self.system_prompt = f"""你是一个路由代理，负责将用户请求路由到合适的专业代理处理。
            
可用的代理:
{agent_descriptions}

分析用户的请求，仅回复最合适的代理名称。
不要添加任何解释或额外文本。"""
        else:
            self.system_prompt = system_prompt

    async def agenerate(
        self, query: str, messages: Optional[List[BaseMessage]] = None, **kwargs
    ) -> AgentResponse:
        """
        路由请求到合适的代理并返回其响应。
        """
        # 确保MCP客户端已设置（如果有）
        await self._setup_mcp_client()

        try:
            # 创建路由提示
            prompt = ChatPromptTemplate.from_messages(
                [("system", self.system_prompt), ("human", "{query}")]
            )

            # 获取合适的代理
            prompt_value = await prompt.ainvoke({"query": query})
            routing_response = await self.model.ainvoke(prompt_value)
            agent_name = routing_response.content.strip().lower()

            # 查找代理
            selected_agent = None
            for name, agent in self.agents.items():
                if name.lower() == agent_name:
                    selected_agent = agent
                    break

            # 如果未找到代理，使用第一个作为默认
            if selected_agent is None:
                if self.agents:
                    selected_agent = next(iter(self.agents.values()))
                else:
                    return AgentResponse(
                        content="无可用代理处理您的请求。",
                        metadata={
                            "agent_name": self.name,
                            "error": "no_agents_available",
                        },
                    )

            # 路由到选定的代理
            agent_response = await selected_agent.agenerate(query, messages, **kwargs)

            # 添加路由元数据
            agent_response.metadata["router"] = self.name
            agent_response.metadata["routed_to"] = selected_agent.name

            return agent_response
        finally:
            # 清理MCP客户端资源
            if self.mcp_servers:
                await self._cleanup_mcp_client()
