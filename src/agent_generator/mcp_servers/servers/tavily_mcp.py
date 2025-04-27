"""
Tavily MCP服务器

一个使用Python实现的Tavily MCP服务器，提供网络搜索和内容提取功能。
"""

import os
import sys
import requests
from typing import Dict, List, Any, Optional, Union
from mcp.server.fastmcp import FastMCP
import logging
# 将上一级文件夹添加到sys.path，确保可以找到uv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 初始化 FastMCP server
mcp = FastMCP("tavily", keep_alive=True)

import dotenv
dotenv.load_dotenv()

# 配置日志
if os.environ.get("LOGGING_MODE") == "off":
    logging.basicConfig(level=logging.WARNING)
    # logging.getLogger("httpx").setLevel(logging.WARNING)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tavily-mcp-server")


# 服务器配置信息
SERVER_NAME = "tavily"
SERVER_CONFIG = {
    "name": SERVER_NAME,
    "command": "uv",
    "args": [
        "--directory",
        os.path.dirname(os.path.abspath(__file__)),
        "run",
        os.path.basename(__file__),
    ],
    "transport": "stdio",
    "description": "Tavily搜索和内容提取MCP服务器",
}

# Tavily API配置
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"


class TavilyAPIError(Exception):
    """Tavily API调用错误"""
    pass


def get_headers():
    """获取API请求头"""
    if not TAVILY_API_KEY:
        raise ValueError("未设置TAVILY_API_KEY环境变量")
    
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TAVILY_API_KEY}"
    }


@mcp.tool(name="tavily_search", description="使用Tavily搜索引擎执行网络搜索")
async def tavily_search(
    query: str,
    search_depth: str = "basic",
    topic: str = "general",
    max_results: int = 5,
    time_range: Optional[str] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_answer: Union[bool, str] = False,
    include_images: bool = False,
    include_image_descriptions: bool = False,
    include_raw_content: bool = False,
    chunks_per_source: Optional[int] = None,
) -> Dict[str, Any]:
    """使用Tavily API执行网络搜索。

    Args:
        query: 搜索查询关键词
        search_depth: 搜索深度，可以是 "basic" 或 "advanced"
        topic: 搜索主题类别，可以是 "general" 或 "news"
        max_results: 返回的最大结果数量，范围在0-20之间
        time_range: 搜索时间范围，例如 "day", "week", "month", "year" 或简写 "d", "w", "m", "y"
        include_domains: 要包含在搜索结果中的域名列表
        exclude_domains: 要排除在搜索结果外的域名列表
        include_answer: 是否包含基于搜索结果生成的答案，可以是布尔值或 "basic"/"advanced" 字符串
        include_images: 是否包含与查询相关的图片
        include_image_descriptions: 是否包含图片描述
        include_raw_content: 是否包含每个搜索结果的原始内容
        chunks_per_source: 从每个源获取的内容块数量（仅在search_depth为advanced时可用）

    Returns:
        包含搜索结果的字典

    Raises:
        TavilyAPIError: 当API调用失败时
    """
    try:
        payload = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
            "include_raw_content": include_raw_content,
        }

        # 添加可选参数
        if time_range:
            payload["time_range"] = time_range
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        if chunks_per_source and search_depth == "advanced" and 1 <= chunks_per_source <= 3:
            payload["chunks_per_source"] = chunks_per_source

        response = requests.post(
            TAVILY_SEARCH_ENDPOINT,
            headers=get_headers(),
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise TavilyAPIError(f"Tavily API调用失败，状态码: {response.status_code}, 消息: {response.text}")

        return response.json()

    except requests.RequestException as e:
        raise TavilyAPIError(f"Tavily搜索请求失败: {str(e)}")


@mcp.tool(name="tavily_news_search", description="使用Tavily搜索引擎执行新闻搜索")
async def tavily_news_search(
    query: str,
    days: int = 7,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_answer: Union[bool, str] = False,
    include_images: bool = False,
) -> Dict[str, Any]:
    """使用Tavily API执行新闻搜索。

    Args:
        query: 搜索查询关键词
        days: 从当前日期往前推的天数，仅在新闻搜索中可用
        max_results: 返回的最大结果数量，范围在0-20之间
        include_domains: 要包含在搜索结果中的域名列表
        exclude_domains: 要排除在搜索结果外的域名列表
        include_answer: 是否包含基于搜索结果生成的答案，可以是布尔值或 "basic"/"advanced" 字符串
        include_images: 是否包含与查询相关的图片

    Returns:
        包含新闻搜索结果的字典

    Raises:
        TavilyAPIError: 当API调用失败时
    """
    try:
        payload = {
            "query": query,
            "topic": "news",
            "days": days,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_images": include_images,
        }

        # 添加可选参数
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        response = requests.post(
            TAVILY_SEARCH_ENDPOINT,
            headers=get_headers(),
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise TavilyAPIError(f"Tavily API调用失败，状态码: {response.status_code}, 消息: {response.text}")

        return response.json()

    except requests.RequestException as e:
        raise TavilyAPIError(f"Tavily新闻搜索请求失败: {str(e)}")


@mcp.tool(name="tavily_extract", description="使用Tavily API从网页提取内容")
async def tavily_extract(
    urls: Union[str, List[str]],
    include_images: bool = False,
    extract_depth: str = "basic",
) -> Dict[str, Any]:
    """使用Tavily API从指定的URL提取内容。

    Args:
        urls: 要提取内容的URL或URL列表（最多20个URL）
        include_images: 是否包含从URL提取的图片
        extract_depth: 提取深度，可以是 "basic" 或 "advanced"
                      "advanced"提取可以检索更多数据，包括表格和嵌入内容

    Returns:
        包含提取内容的字典

    Raises:
        TavilyAPIError: 当API调用失败时
        ValueError: 当URL列表超过20个时
    """
    try:
        # 检查URL数量
        if isinstance(urls, list) and len(urls) > 20:
            raise ValueError("最多只能提供20个URL进行内容提取")

        payload = {
            "urls": urls,
            "include_images": include_images,
            "extract_depth": extract_depth,
        }

        response = requests.post(
            TAVILY_EXTRACT_ENDPOINT,
            headers=get_headers(),
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise TavilyAPIError(f"Tavily API调用失败，状态码: {response.status_code}, 消息: {response.text}")

        return response.json()

    except requests.RequestException as e:
        raise TavilyAPIError(f"Tavily内容提取请求失败: {str(e)}")


@mcp.tool(name="get_tavily_api_status", description="检查Tavily API密钥和连接状态")
async def get_tavily_api_status() -> Dict[str, Any]:
    """检查Tavily API密钥和连接状态。

    Returns:
        包含API状态信息的字典
    """
    try:
        if not TAVILY_API_KEY:
            return {
                "status": "error",
                "message": "未设置TAVILY_API_KEY环境变量",
                "api_key_set": False,
            }

        # 尝试进行一个简单的搜索请求来验证API密钥
        response = requests.post(
            TAVILY_SEARCH_ENDPOINT,
            headers=get_headers(),
            json={"query": "test", "max_results": 1},
            timeout=60,
        )

        if response.status_code == 200:
            return {
                "status": "success",
                "message": "Tavily API连接正常",
                "api_key_set": True,
            }
        elif response.status_code == 401:
            return {
                "status": "error",
                "message": "Tavily API密钥无效",
                "api_key_set": True,
            }
        else:
            return {
                "status": "error",
                "message": f"Tavily API连接异常，状态码: {response.status_code}",
                "api_key_set": True,
            }

    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"Tavily API连接失败: {str(e)}",
            "api_key_set": bool(TAVILY_API_KEY),
        }


if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run(transport="stdio") 