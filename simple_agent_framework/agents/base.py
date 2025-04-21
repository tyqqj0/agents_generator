from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
import asyncio

class AgentResponse(BaseModel):
    """Unified agent response format"""
    content: str
    raw_response: Any = None  # Original response object
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class Agent:
    """Base class for all agents, providing a unified interface"""
    
    def __init__(self, name: str, model: Optional[BaseChatModel] = None):
        self.name = name
        self.model = model
        
    async def agenerate(self, 
                      query: str, 
                      messages: Optional[List[BaseMessage]] = None, 
                      **kwargs) -> AgentResponse:
        """Asynchronously generate a response"""
        raise NotImplementedError
    
    def generate(self, 
                query: str, 
                messages: Optional[List[BaseMessage]] = None, 
                **kwargs) -> AgentResponse:
        """Synchronously generate a response"""
        return asyncio.run(self.agenerate(query, messages, **kwargs))
    
    def __call__(self, query: str, **kwargs) -> str:
        """Convenient calling method, directly returns content string"""
        return self.generate(query, **kwargs).content