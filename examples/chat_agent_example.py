"""
Chat Agent Example

This example demonstrates how to use the ChatAgent for basic conversation.
"""

import os
import asyncio
from langchain_openai import ChatOpenAI
from simple_agent_framework import ChatAgent

# Load API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

async def main():
    # Create an LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create a chat agent
    chat_agent = ChatAgent(
        name="travel_assistant",
        model=llm,
        system_prompt="You are a helpful travel assistant. You provide advice about destinations, accommodations, and activities."
    )
    
    # Generate a response asynchronously
    response = await chat_agent.agenerate(
        "I'm planning a trip to Japan for 10 days. Where should I visit?"
    )
    
    print(f"Response: {response.content}")
    print(f"Metadata: {response.metadata}")
    
    # You can also use the agent with the synchronous interface
    print("\nSecond query (synchronous):")
    result = chat_agent.generate("What's the best time of year to visit Kyoto?")
    print(f"Response: {result.content}")
    
    # Or even more simply with the call syntax
    print("\nThird query (call syntax):")
    answer = chat_agent("What should I know about Japanese etiquette?")
    print(f"Response: {answer}")

if __name__ == "__main__":
    asyncio.run(main()) 