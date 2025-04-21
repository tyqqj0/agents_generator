from setuptools import setup, find_packages

setup(
    name="simple_agent_framework",
    version="0.1.0",
    description="A simple framework for building and using different types of AI agents",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "pydantic>=2.0.0",
        "mcp>=0.1.0",
    ],
    python_requires=">=3.8",
)
