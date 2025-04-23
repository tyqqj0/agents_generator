# -*- coding: utf-8 -*-
"""
@File    :   setup.py
@Time    :   2025/04/23 14:35:43
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""




from setuptools import setup, find_packages

setup(
    name="agent_generator",
    version="0.1.0",
    description="智能代理生成和管理框架",
    author="AI开发者",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-openai",
        "dotenv",
    ],
    python_requires=">=3.8",
) 