from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from .base import Agent, AgentResponse
from ..connectors.mcp import MCPConnector
from ..templates.prompts import DEFAULT_TOOL_PROMPT

class ToolAgent(Agent):
    """Agent that uses MCP tools"""
    
    def __init__(self, 
                name: str, 
                model,
                mcp_config: Dict[str, Any],
                system_prompt: Optional[str] = None,
                temperature: float = 0.7):
        super().__init__(name, model)
        self.system_prompt = system_prompt or DEFAULT_TOOL_PROMPT
        self.temperature = temperature
        self.connector = MCPConnector(name, mcp_config)
        
    async def agenerate(self, 
                      query: str, 
                      messages: Optional[List[BaseMessage]] = None,
                      **kwargs) -> AgentResponse:
        """Generate a response, including tool calls"""
        # Initialize MCP connector
        await self.connector.connect()
        
        try:
            # Get tool list
            tools = await self.connector.list_tools()
            
            # Prepare message history
            message_history = messages or []
            
            # Add current query
            message_history.append(HumanMessage(content=query))
            
            # Bind tools to model
            model_with_tools = self.model.bind_tools(tools)
            
            # Generate response
            response = await model_with_tools.ainvoke(
                message_history,
                temperature=self.temperature,
                **kwargs
            )
            
            # Handle tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_results = []
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    
                    # Call tool
                    tool_result = await self.connector.execute_tool(tool_name, **tool_args)
                    
                    # Record tool result
                    tool_results.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call.get("id"))
                    )
                    
                # Update message history
                message_history.extend(tool_results)
                
                # Generate final response
                final_response = await self.model.ainvoke(message_history)
                
                return AgentResponse(
                    content=final_response.content,
                    raw_response=final_response,
                    metadata={
                        "agent_name": self.name,
                        "tool_calls": response.tool_calls,
                        "tool_results": tool_results
                    }
                )
            
            return AgentResponse(
                content=response.content,
                raw_response=response,
                metadata={"agent_name": self.name}
            )
        finally:
            # Ensure disconnection
            await self.connector.disconnect() 