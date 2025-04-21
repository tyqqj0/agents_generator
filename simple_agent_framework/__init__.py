from .agents.base import Agent, AgentResponse
from .agents.chat_agent import ChatAgent
from .agents.tool_agent import ToolAgent
from .agents.router_agent import RouterAgent
from .connectors.mcp import MCPConnector
from .templates.prompts import (
    DEFAULT_CHAT_PROMPT,
    DEFAULT_TOOL_PROMPT,
    DEFAULT_RESEARCH_PROMPT,
    DEFAULT_CODE_PROMPT,
)

__version__ = "0.1.0"
