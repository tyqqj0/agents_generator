from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from .base import BaseAgent, AgentResponse
from ..templates.prompts import DEFAULT_CHAT_PROMPT


class ChatAgent(BaseAgent):
    """简单聊天代理，使用LLM生成响应"""

    def __init__(
        self,
        name: str,
        model,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        初始化聊天代理

        Args:
            name: 代理名称
            model: 使用的语言模型
            system_prompt: 系统提示，定义代理行为
            temperature: 温度参数，控制输出随机性
            tools: 预定义工具列表
            mcp_servers: MCP服务器配置
        """
        super().__init__(name, model, tools, mcp_servers)
        self.system_prompt = system_prompt or DEFAULT_CHAT_PROMPT
        self.temperature = temperature

    async def agenerate(
        self, query: str, messages: Optional[List[BaseMessage]] = None, **kwargs
    ) -> AgentResponse:
        """生成响应"""
        # 确保MCP客户端已设置（如果有）
        await self._setup_mcp_client()

        try:
            # 准备消息历史
            message_history = messages or []

            # 添加当前查询
            message_history.append(HumanMessage(content=query))

            # 创建提示
            prompt = ChatPromptTemplate.from_messages(
                [("system", self.system_prompt), ("placeholder", "{messages}")]
            )

            # 准备提示输入
            prompt_input = {"messages": message_history}

            # 获取提示值
            prompt_value = await prompt.ainvoke(prompt_input)

            # 如果有工具，则绑定到模型
            model = self.model
            if self.tools:
                model = self.model.bind_tools(self.tools)

            # 调用模型
            response = await model.ainvoke(
                prompt_value, temperature=self.temperature, **kwargs
            )

            # 检查是否有工具调用
            if hasattr(response, "tool_calls") and response.tool_calls:
                # 如果有工具调用但没有设置工具，发出警告
                if not self.tools:
                    return AgentResponse(
                        content="抱歉，似乎我需要使用工具来回答这个问题，但我没有工具权限。请尝试使用ToolAgent。",
                        metadata={
                            "agent_name": self.name,
                            "error": "no_tools_available",
                        },
                    )

                # 这里可以添加对工具调用的处理，但通常由ToolAgent负责
                # 简单起见，这里返回带有工具调用信息的响应
                return AgentResponse(
                    content=response.content,
                    raw_response=response,
                    metadata={
                        "agent_name": self.name,
                        "has_tool_calls": True,
                        "tool_calls": response.tool_calls,
                    },
                )

            return AgentResponse(
                content=response.content,
                raw_response=response,
                metadata={"agent_name": self.name},
            )
        finally:
            # 清理MCP客户端资源
            if self.mcp_servers:
                await self._cleanup_mcp_client()
