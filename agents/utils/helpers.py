from typing import Dict, Any, Optional, Union, List
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def load_llm(model_name: str, **kwargs) -> BaseChatModel:
    """
    Load a language model by name.
    
    Args:
        model_name: String in format "provider/model_name", e.g. "openai/gpt-4"
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        BaseChatModel: A language model instance
    """
    provider, model = model_name.split("/", 1)
    
    if provider.lower() == "openai":
        return ChatOpenAI(model=model, **kwargs)
    elif provider.lower() == "anthropic":
        return ChatAnthropic(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
        
def format_messages_for_prompt(messages: List[Any]) -> str:
    """
    Format a list of messages for inclusion in a prompt.
    
    Args:
        messages: List of messages (dict or BaseMessage objects)
        
    Returns:
        str: Formatted message string
    """
    formatted = []
    
    for msg in messages:
        # Handle dict format
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        # Handle BaseMessage format
        else:
            role = type(msg).__name__.replace("Message", "").lower()
            content = msg.content
            formatted.append(f"{role}: {content}")
            
    return "\n".join(formatted) 