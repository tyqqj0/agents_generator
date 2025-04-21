"""
Router Agent Example

This example demonstrates how to use the RouterAgent to route queries to specialized agents.
"""

import os
import asyncio
from langchain_openai import ChatOpenAI
from simple_agent_framework import RouterAgent, ChatAgent, ToolAgent

# Load API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

async def main():
    # Create an LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3
    )
    
    # Create a ChatAgent for general questions
    general_agent = ChatAgent(
        name="general",
        model=llm,
        system_prompt="You are a helpful general assistant that answers questions on a wide range of topics."
    )
    
    # Create a ChatAgent for travel advice
    travel_agent = ChatAgent(
        name="travel",
        model=llm,
        system_prompt="You are a travel expert who provides detailed advice about destinations, accommodations, and activities."
    )
    
    # Create a ChatAgent for coding help
    coding_agent = ChatAgent(
        name="coding",
        model=llm,
        system_prompt="You are a programming expert who helps users write, debug, and understand code."
    )
    
    # Create a router agent with a custom system prompt
    router = RouterAgent(
        name="router",
        model=llm,
        agents={
            "general": general_agent,
            "travel": travel_agent,
            "coding": coding_agent
        },
        system_prompt="""You are a router agent responsible for directing user requests to the appropriate specialized agent.
            
Available agents:
- general: For general questions on various topics
- travel: For travel-related questions and advice
- coding: For programming and development questions

Analyze the user's request and respond ONLY with the name of the most appropriate agent.
Do not add any explanation or additional text."""
    )
    
    # Test with different types of queries
    queries = [
        "What is the capital of France?",
        "I'm planning a trip to Japan. What should I see in Tokyo?",
        "How do I write a recursive function in Python?",
    ]
    
    for query in queries:
        print(f"\n\nQuery: {query}")
        response = await router.agenerate(query)
        print(f"Routed to: {response.metadata.get('routed_to')}")
        print(f"Response: {response.content}")

if __name__ == "__main__":
    asyncio.run(main()) 