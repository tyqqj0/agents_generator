from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from .base import Agent, AgentResponse
from ..templates.prompts import DEFAULT_CHAT_PROMPT

class ChatAgent(Agent):
    """Simple chat agent that uses an LLM to generate responses"""
    
    def __init__(self, 
                name: str, 
                model,
                system_prompt: Optional[str] = None,
                temperature: float = 0.7):
        super().__init__(name, model)
        self.system_prompt = system_prompt or DEFAULT_CHAT_PROMPT
        self.temperature = temperature
        
    async def agenerate(self, 
                      query: str, 
                      messages: Optional[List[BaseMessage]] = None,
                      **kwargs) -> AgentResponse:
        """Generate a response"""
        # Prepare message history
        message_history = messages or []
        
        # Add current query
        message_history.append(HumanMessage(content=query))
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{messages}")
        ])
        
        # Prepare prompt input
        prompt_input = {"messages": message_history}
        
        # Get prompt value
        prompt_value = await prompt.ainvoke(prompt_input)
        
        # Call model
        response = await self.model.ainvoke(
            prompt_value,
            temperature=self.temperature,
            **kwargs
        )
        
        return AgentResponse(
            content=response.content,
            raw_response=response,
            metadata={"agent_name": self.name}
        ) 