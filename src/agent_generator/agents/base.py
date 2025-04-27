# -*- coding: utf-8 -*-
"""
@File    :   base.py
@Time    :   2025/04/23 14:33:17
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





from typing import List, Dict, Any, Optional, Union, Set
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import asyncio
from contextlib import AsyncExitStack
import base64

# 新增导入MCP工具相关库
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from abc import ABC, abstractmethod
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph

# 新增可视化相关的导入
try:
    from IPython.display import Image, display
    IPYTHON_DISPLAY_AVAILABLE = True
except ImportError:
    IPYTHON_DISPLAY_AVAILABLE = False




class BaseAgent(ABC):
    """所有代理的基类，提供统一接口"""

    def __init__(
        self,
        name: str,
        model: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
        memory: Optional[BaseChatModel] = None,
        **kwargs,
    ):
        """
        初始化代理

        Args:
            name: 代理名称
            model: 使用的语言模型
            tools: 预定义的工具列表
            mcp_servers: 多MCP服务器配置，格式为 {"server_name": {"command": "...", "args": [...], "transport": "..."}}
            memory: 用于记忆的模型(代支持)
        """
        self.name = name
        if not isinstance(model, BaseChatModel):
            raise ValueError("model必须是BaseChatModel类型")
        self.model = model
        self.tools = tools or []
        self.mcp_servers = mcp_servers or {}
        self.memory = memory
        self.mcp_client = None
        self.agent = None
        self._graph_initialized = False # 标记图是否已初始化
    
    # 后初始化，在进入上下文管理器时调用，确保工具绑定
    def _post_init(self):
        if not self._graph_initialized:
            self.agent = self._create_agent()
            self._graph_initialized = True
        # print(f"agent: {self.agent}")
    @abstractmethod
    def _create_agent(self) -> CompiledGraph:
        """创建代理, 子类必须实现"""
        pass

    def _get_messages(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[BaseMessage]:
        """
        获取消息列表，支持文本和多模态输入
        
        Args:
            input_data: 输入数据
                - 字符串: 作为纯文本消息处理
                - 字典: 包含多模态内容的消息，可以是图像或其他媒体类型
                - 列表: 多个内容块组成的消息
        
        Returns:
            List[BaseMessage]: 消息列表
        """
        # 如果是字符串，作为纯文本消息处理
        if isinstance(input_data, str):
            return [HumanMessage(content=input_data)]
        
        # 如果是字典或列表，作为多模态消息处理
        elif isinstance(input_data, (dict, list)):
            # 将单个字典转为列表处理
            if isinstance(input_data, dict):
                content = [input_data]
            else:
                content = input_data
                
            # 检查内容结构是否符合多模态格式
            processed_content = []
            for item in content:
                if isinstance(item, dict) and "type" in item:
                    processed_content.append(item)
                elif isinstance(item, str): # 容错处理，将字符串包装成text类型
                    processed_content.append({"type": "text", "text": item})
                else:
                    # 对于不确定类型，尝试转为字符串
                    processed_content.append({"type": "text", "text": str(item)})

            return [HumanMessage(content=processed_content)]
        
        # 默认情况，尝试转为字符串处理
        return [HumanMessage(content=str(input_data))]
    
    def _get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """获取适合LLM使用的工具定义，子类可以覆盖以自定义工具构建逻辑"""
        if not self.tools:
            return []
        return [tool.dict() for tool in self.tools if hasattr(tool, "dict")]

    async def agenerate(self, input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]], image_path: Optional[str] = None, image_url: Optional[str] = None, messages: Optional[List[BaseMessage]] = None, **kwargs):
        """
        异步生成回复，支持文本和图像输入
        
        Args:
            input_data: 输入数据
                - 字符串: 纯文本提示
                - 字典: 包含多模态内容的消息数据
                - 列表: 多个内容块组成的消息
            image_path: 可选，本地图像文件路径
            image_url: 可选，图像URL
            messages: 可选，已有的消息历史
            **kwargs: 其他参数
        
        Returns:
            模型响应
        """
        # 确保代理图已初始化
        self._post_init()

        # 如果提供了图像，构建多模态输入
        final_input_data = input_data
        if image_path or image_url:
            # 构建基本文本内容
            base_text = input_data if isinstance(input_data, str) else str(input_data)
            content = [{"type": "text", "text": base_text}]
            
            # 如果有本地图像路径，读取并编码为base64
            if image_path:
                try:
                    with open(image_path, "rb") as img_file:
                        image_data = base64.b64encode(img_file.read()).decode("utf-8")
                        content.append({
                            "type": "image_url", # OpenAI vision model期望的类型是 image_url
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"} # data URL 格式
                        })
                except Exception as e:
                    raise ValueError(f"读取图像文件失败: {str(e)}")
            
            # 如果有图像URL
            elif image_url:
                 content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            # 使用多模态内容
            final_input_data = content

        # 如果没有绑定工具，则绑定工具
        if not self.tools and self.mcp_servers:
            await self._setup_mcp_client()

            tool_names = [getattr(tool, "name", str(tool)) for tool in self.tools]
            # print(f"绑定工具: {tool_names}")  # 调试信息

        # 准备消息
        prepared_messages = messages if messages is not None else []
        prepared_messages.extend(self._get_messages(final_input_data))

        # 异步执行代理
        # print(f"Invoking agent with messages: {prepared_messages}") # 调试信息
        response = await self.agent.ainvoke({"messages": prepared_messages})

        
        return response


    async def _setup_mcp_client(self):
        """设置MCP客户端并加载工具"""
        # print(f"设置MCP客户端并加载工具: {self.mcp_servers}")
        if self.mcp_servers and not self.mcp_client: # 避免重复设置
            # 确保mcp_servers是一个字典
            if isinstance(self.mcp_servers, dict):
                try:
                    self.mcp_client = MultiServerMCPClient(self.mcp_servers)
                    await self.mcp_client.__aenter__()
                    
                    mcp_tools_result = None
                    if hasattr(self.mcp_client, "get_tools"):
                        mcp_tools_result = self.mcp_client.get_tools()
                    else:
                        mcp_tools_result = load_mcp_tools(self.mcp_client)

                    if asyncio.iscoroutine(mcp_tools_result):
                        mcp_tools = await mcp_tools_result
                    else:
                        mcp_tools = mcp_tools_result

                    if hasattr(mcp_tools, "__iter__"):
                        # 合并工具时去重
                        existing_tool_names = {getattr(t, 'name', str(t)) for t in self.tools}
                        new_tools = [tool for tool in mcp_tools if getattr(tool, 'name', str(tool)) not in existing_tool_names]
                        self.tools.extend(new_tools)
                    else:
                        raise TypeError(
                            f"MCP工具加载函数应该返回可迭代对象，当前类型为: {type(mcp_tools)}"
                        )
                except Exception as e:
                    print(f"设置 MCP 客户端时出错: {e}") # 打印错误信息
                    if self.mcp_client:
                        try:
                            await self.mcp_client.__aexit__(type(e), e, None)
                        except Exception as cleanup_exc:
                             print(f"清理 MCP 客户端时出错: {cleanup_exc}") # 打印清理错误
                    self.mcp_client = None
                    # 不在此处重新抛出异常，允许程序继续运行（可能没有MCP工具）
            else:
                raise TypeError(
                    f"mcp_servers应该是字典类型，当前类型为: {type(self.mcp_servers)}"
                )

    async def _cleanup_mcp_client(self):
        """清理MCP客户端资源"""
        if self.mcp_client:
            try:
                await self.mcp_client.__aexit__(None, None, None)
            except Exception as e:
                print(f"清理 MCP 客户端时出错: {e}") # 打印错误信息
            finally:
                self.mcp_client = None

    def visualize(self, save_path: Optional[str] = None):
        """
        可视化代理的计算图。
        需要安装 graphviz 和 python-pygraphviz / python-graphviz。
        在Jupyter环境或类似环境中会尝试直接显示图像。
        """
        if not self._graph_initialized:
            print("错误：代理图尚未初始化。请先调用 agenerate 或进入异步上下文。")
            return

        if not self.agent:
            print("错误：代理对象 (self.agent) 未被创建。")
            return
            
        if not hasattr(self.agent, 'get_graph'):
            print("错误：当前代理对象不支持 get_graph 方法。")
            return

        graph = self.agent.get_graph()

        if not hasattr(graph, 'draw_mermaid_png'):
             print("错误：代理图对象不支持 draw_mermaid_png 方法。请确保安装了必要的依赖（如 mermaid-cli）。")
             return

        if IPYTHON_DISPLAY_AVAILABLE:
            try:
                # 生成并显示 Mermaid 图
                png_data = graph.draw_mermaid_png()
                
                # 如果 save_path 为 None，则将 PNG 数据保存为 agent_graph.png
                if save_path is None:
                    save_path = 'agent_graph.png'
                # 将 PNG 数据保存到文件
                if save_path:
                    with open(save_path, 'wb') as f:
                        f.write(png_data)
                    print(f"代理图已成功渲染并保存为 {save_path}")
                else:
                    print("代理图保存路径为空，不保存")
            except Exception as e:
                print(f"渲染代理图时发生错误: {e}")
                print("请确保已安装 graphviz 和 python-pygraphviz / python-graphviz 以及 mermaid-cli (npm install -g @mermaid-js/mermaid-cli)。")
        else:
            print("警告：IPython.display 不可用。无法在当前环境中直接显示图像。")
            print("您可以尝试手动调用 graph.draw_mermaid_png() 并将结果保存到文件。")


    async def __aenter__(self):
        """异步上下文管理支持"""
        await self._setup_mcp_client()
        self._post_init() # 确保图在进入上下文时初始化
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文退出时清理资源"""
        await self._cleanup_mcp_client()
