from typing import Dict, Any, List, Optional
from mcp import stdio_client
import asyncio

class MCPConnector:
    """MCP tool connector"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.session = None
        
    async def connect(self):
        """Connect to MCP server"""
        if self.session is not None:
            return
            
        command = self.config.get("command")
        args = self.config.get("args", [])
        
        self.session = await stdio_client(command, *args)
        return self
        
    async def disconnect(self):
        """Disconnect from MCP"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Get available tools list and convert to OpenAI format"""
        if not self.session:
            await self.connect()
            
        mcp_tools = await self.session.tools.list()
        
        # Convert to OpenAI tool format
        openai_tools = []
        for tool in mcp_tools:
            openai_tool = {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
            openai_tools.append(openai_tool)
            
        return openai_tools
        
    async def execute_tool(self, tool_name: str, **params) -> Any:
        """Execute a tool"""
        if not self.session:
            await self.connect()
            
        return await self.session.tools.call(tool_name, **params) 