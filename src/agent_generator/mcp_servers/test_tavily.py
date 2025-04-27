# -*- coding: utf-8 -*-
"""
测试Tavily MCP服务器的客户端

提供自动测试和交互式测试功能，用于验证Tavily MCP服务器的搜索和提取功能。
"""

import asyncio
import sys
import os
import json
from typing import Optional, Dict, Any, List, Union
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dotenv


dotenv.load_dotenv()

# 检查Tavily API密钥是否已设置
if not os.environ.get("TAVILY_API_KEY"):
    print(os.environ.keys())
    raise ValueError("Tavily API密钥未设置")


def process_mcp_content(content):
    """处理MCP返回的内容，将TextContent转换为Python对象。

    Args:
        content: MCP返回的内容

    Returns:
        处理后的Python对象
    """
    # 如果是列表，处理每个元素
    if isinstance(content, list):
        return [process_mcp_content(item) for item in content]
    
    # 如果有text属性（TextContent对象）
    if hasattr(content, "text"):
        try:
            # 尝试解析为JSON
            return json.loads(content.text)
        except (json.JSONDecodeError, AttributeError):
            # 如果不是有效的JSON，直接返回文本
            return content.text
    
    # 其他情况直接返回
    return content


class TavilyTestClient:
    def __init__(self):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None  # 用于与 MCP 服务器通信的会话
        self.exit_stack = AsyncExitStack()  # 用于管理资源
        self.tools = []  # 存储可用工具列表

    async def connect_to_server(self, server_script_path: str):
        """连接到Tavily MCP服务器

        Args:
            server_script_path: 服务器脚本的路径
        """
        # print(server_script_path)
        is_python = server_script_path.endswith(".py")
        if not is_python:
            raise ValueError("服务器脚本必须是 .py 文件")

        print(f"正在连接到服务器: {server_script_path}")

        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env={"PYTHONIOENCODING": "utf-8"},  # 确保Python进程使用UTF-8编码
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()

            # 列出可用的工具
            response = await self.session.list_tools()
            self.tools = response.tools
            print("\n已连接到服务器，可用工具包括：")
            for tool in self.tools:
                print(f" - {tool.name}: {tool.description}")

            return True

        except Exception as e:
            print(f"连接服务器时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def test_api_status(self):
        """测试Tavily API状态"""
        if not self.session:
            print("错误: 未连接到服务器")
            return False

        print("\n测试API状态:")
        try:
            result = await self.session.call_tool("get_tavily_api_status", {})
            status_data = process_mcp_content(result.content)
            
            # 打印状态信息
            print(f"API状态: {json.dumps(status_data, ensure_ascii=False, indent=2)}")
            
            # 处理可能的列表结果
            if isinstance(status_data, list) and status_data:
                status_data = status_data[0]  # 获取列表中的第一个元素
            
            # 检查API密钥是否已设置
            api_key_set = status_data.get("api_key_set", False)
            status = status_data.get("status", "unknown")

            if api_key_set and status == "success":
                print("✓ Tavily API连接正常")
                return True
            else:
                error_msg = status_data.get("message", "未知错误")
                print(f"✗ Tavily API状态异常: {error_msg}")
                return False

        except Exception as e:
            print(f"✗ 测试API状态时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def test_tavily_search(self):
        """测试Tavily搜索功能"""
        if not self.session:
            print("错误: 未连接到服务器")
            return False

        print("\n测试Tavily搜索功能:")
        
        # 测试一般搜索
        try:
            print("测试一般搜索...")
            search_params = {
                "query": "人工智能最新发展",
                "max_results": 2,
                "search_depth": "basic"
            }
            
            result = await self.session.call_tool("tavily_search", search_params)
            search_results = process_mcp_content(result.content)
            
            # 检查结果
            if "results" in search_results and isinstance(search_results["results"], list):
                print(f"✓ 一般搜索成功，获取到 {len(search_results['results'])} 条结果")
                print(f"  查询: {search_results.get('query', '')}")
                print(f"  响应时间: {search_results.get('response_time', 0)} 秒")
                
                # 打印第一条结果的摘要
                if search_results["results"]:
                    first_result = search_results["results"][0]
                    print(f"\n  第一条结果标题: {first_result.get('title', '无标题')}")
                    content = first_result.get('content', '')
                    print(f"  内容摘要: {content[:100]}..." if len(content) > 100 else content)
            else:
                print(f"✗ 一般搜索失败，未获取到有效结果")
                return False
                
        except Exception as e:
            print(f"✗ 测试一般搜索时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        # 测试新闻搜索
        try:
            print("\n测试新闻搜索...")
            news_params = {
                "query": "人工智能新闻",
                "days": 3,
                "max_results": 2
            }
            
            result = await self.session.call_tool("tavily_news_search", news_params)
            news_results = process_mcp_content(result.content)
            
            # 检查结果
            if "results" in news_results and isinstance(news_results["results"], list):
                print(f"✓ 新闻搜索成功，获取到 {len(news_results['results'])} 条结果")
                # 打印第一条结果的摘要
                if news_results["results"]:
                    first_result = news_results["results"][0]
                    print(f"  第一条新闻标题: {first_result.get('title', '无标题')}")
                    if "published_date" in first_result:
                        print(f"  发布日期: {first_result['published_date']}")
            else:
                print(f"✗ 新闻搜索失败，未获取到有效结果")
                return False
                
        except Exception as e:
            print(f"✗ 测试新闻搜索时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        return True

    async def test_tavily_extract(self):
        """测试Tavily内容提取功能"""
        if not self.session:
            print("错误: 未连接到服务器")
            return False

        print("\n测试Tavily内容提取功能:")
        
        try:
            # 使用一个知名的网站进行测试
            extract_params = {
                "urls": "https://www.example.com",
                "include_images": False,
                "extract_depth": "basic"
            }
            
            result = await self.session.call_tool("tavily_extract", extract_params)
            extract_results = process_mcp_content(result.content)
            
            # 检查结果
            if "results" in extract_results and isinstance(extract_results["results"], list):
                print(f"✓ 内容提取成功，获取到 {len(extract_results['results'])} 个URL的内容")
                
                # 打印提取内容的摘要
                if extract_results["results"]:
                    first_result = extract_results["results"][0]
                    print(f"  URL: {first_result.get('url', '未知URL')}")
                    raw_content = first_result.get('raw_content', '')
                    print(f"  内容摘要: {raw_content[:100]}..." if len(raw_content) > 100 else raw_content)
                    
                # 检查是否有失败的提取
                if extract_results.get("failed_results", []):
                    print(f"  警告: {len(extract_results['failed_results'])} 个URL提取失败")
                    for failed in extract_results["failed_results"]:
                        print(f"    - {failed.get('url', '未知URL')}: {failed.get('error', '未知错误')}")
            else:
                print(f"✗ 内容提取失败，未获取到有效结果")
                return False
                
        except Exception as e:
            print(f"✗ 测试内容提取时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        return True

    async def run_all_tests(self):
        """运行所有测试"""
        if not self.session:
            print("错误: 未连接到服务器")
            return False

        # 首先测试API状态
        api_status_ok = await self.test_api_status()
        if not api_status_ok:
            print("\n⚠️ API状态异常，跳过后续测试")
            return False
            
        # 测试搜索功能
        search_ok = await self.test_tavily_search()
        
        # 测试内容提取功能
        extract_ok = await self.test_tavily_extract()
        
        # 打印总结
        print("\n测试总结:")
        print(f"API状态检查: {'通过' if api_status_ok else '失败'}")
        print(f"搜索功能测试: {'通过' if search_ok else '失败'}")
        print(f"内容提取测试: {'通过' if extract_ok else '失败'}")
        
        overall_result = api_status_ok and search_ok and extract_ok
        print(f"\n总体测试结果: {'通过' if overall_result else '失败'}")
        
        return overall_result

    async def interactive_mode(self):
        """进入交互模式，手动测试Tavily功能"""
        if not self.session:
            print("错误: 未连接到服务器")
            return

        print("\n进入交互模式。")
        print("可用命令:")
        print("1. search <查询> - 执行一般搜索")
        print("2. news <查询> [天数=7] - 执行新闻搜索")
        print("3. extract <URL> - 提取网页内容")
        print("4. status - 检查API状态")
        print("5. help - 显示帮助信息")
        print("6. quit - 退出")

        while True:
            try:
                cmd = input("\n> ").strip()
                parts = cmd.split(maxsplit=2)
                
                if not parts:
                    continue
                    
                action = parts[0].lower()
                
                if action == "quit":
                    break
                    
                elif action == "help":
                    print("可用命令:")
                    print("1. search <查询> - 执行一般搜索")
                    print("2. news <查询> [天数=7] - 执行新闻搜索")
                    print("3. extract <URL> - 提取网页内容")
                    print("4. status - 检查API状态")
                    print("5. help - 显示帮助信息")
                    print("6. quit - 退出")
                    
                elif action == "status":
                    result = await self.session.call_tool("get_tavily_api_status", {})
                    status_data = process_mcp_content(result.content)
                    print(json.dumps(status_data, ensure_ascii=False, indent=2))
                    
                elif action == "search":
                    if len(parts) < 2:
                        print("错误: 缺少查询参数。用法: search <查询>")
                        continue
                        
                    query = parts[1]
                    search_params = {
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True
                    }
                    
                    print(f"正在搜索: {query}...")
                    result = await self.session.call_tool("tavily_search", search_params)
                    search_results = process_mcp_content(result.content)
                    
                    # 处理可能的列表结果
                    if isinstance(search_results, list) and search_results:
                        print(len(search_results))
                        print(search_results)
                        search_results = search_results[0]  # 获取列表中的第一个元素
                        
                    
                    # 如果有生成的答案，先显示
                    if "answer" in search_results and search_results["answer"]:
                        print("\n生成的答案:")
                        print(search_results["answer"])
                        
                    # 显示搜索结果
                    if "results" in search_results and search_results["results"]:
                        print(f"\n搜索结果 (共 {len(search_results['results'])} 条):")
                        for i, item in enumerate(search_results["results"]):
                            print(f"\n结果 {i+1}:")
                            print(f"标题: {item.get('title', '无标题')}")
                            print(f"URL: {item.get('url', '无URL')}")
                            print(f"相关度: {item.get('score', 0)}")
                            print(f"内容: {item.get('content', '无内容')[:200]}..." if len(item.get('content', '')) > 200 else item.get('content', '无内容'))
                    else:
                        print(json.dumps(search_results, indent=4))
                        print("未找到搜索结果")
                        
                elif action == "news":
                    if len(parts) < 2:
                        print("错误: 缺少查询参数。用法: news <查询> [天数=7]")
                        continue
                        
                    query = parts[1]
                    days = 7
                    
                    # 如果指定了天数
                    if len(parts) > 2:
                        try:
                            days = int(parts[2])
                        except ValueError:
                            print(f"错误: 天数必须是整数，将使用默认值 {days}")
                    
                    news_params = {
                        "query": query,
                        "days": days,
                        "max_results": 3
                    }
                    
                    print(f"正在搜索 {days} 天内的新闻: {query}...")
                    result = await self.session.call_tool("tavily_news_search", news_params)
                    news_results = process_mcp_content(result.content)
                    
                    # 显示新闻结果
                    if "results" in news_results and news_results["results"]:
                        print(f"\n新闻结果 (共 {len(news_results['results'])} 条):")
                        for i, item in enumerate(news_results["results"]):
                            print(f"\n新闻 {i+1}:")
                            print(f"标题: {item.get('title', '无标题')}")
                            print(f"发布日期: {item.get('published_date', '未知日期')}")
                            print(f"URL: {item.get('url', '无URL')}")
                            print(f"内容: {item.get('content', '无内容')[:200]}..." if len(item.get('content', '')) > 200 else item.get('content', '无内容'))
                    else:
                        print("未找到新闻结果")
                        
                elif action == "extract":
                    if len(parts) < 2:
                        print("错误: 缺少URL参数。用法: extract <URL>")
                        continue
                        
                    url = parts[1]
                    extract_params = {
                        "urls": url,
                        "include_images": False
                    }
                    
                    print(f"正在提取URL内容: {url}...")
                    result = await self.session.call_tool("tavily_extract", extract_params)
                    extract_results = process_mcp_content(result.content)
                    
                    # 显示提取结果
                    if "results" in extract_results and extract_results["results"]:
                        print(f"\n成功提取 {len(extract_results['results'])} 个URL的内容:")
                        for i, item in enumerate(extract_results["results"]):
                            print(f"\nURL {i+1}: {item.get('url', '未知URL')}")
                            raw_content = item.get('raw_content', '')
                            print(f"内容摘要: {raw_content[:300]}..." if len(raw_content) > 300 else raw_content)
                            if item.get('images'):
                                print(f"提取到 {len(item['images'])} 张图片")
                    
                    # 显示失败的提取
                    if extract_results.get("failed_results"):
                        print(f"\n{len(extract_results['failed_results'])} 个URL提取失败:")
                        for failed in extract_results["failed_results"]:
                            print(f"URL: {failed.get('url', '未知URL')}")
                            print(f"错误: {failed.get('error', '未知错误')}")
                    
                    print(f"\n响应时间: {extract_results.get('response_time', 0)} 秒")
                    
                else:
                    print(f"未知命令: {action}。输入 'help' 查看可用命令。")

            except Exception as e:
                print(f"错误: {str(e)}")
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """清理资源"""
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"清理资源时出错: {str(e)}")


async def main():
    # 设置环境变量确保使用UTF-8编码
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # 获取Tavily服务器脚本路径
    if len(sys.argv) > 1:
        server_path = sys.argv[1]
    else:
        # 默认路径
        # print(sys.argv)
        server_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "servers", "tavily_mcp.py"
        )
    print(server_path)
    client = TavilyTestClient()
    try:
        connected = await client.connect_to_server(server_path)
        if not connected:
            print("无法连接到服务器，退出")
            return

        # 运行测试模式或交互模式
        if len(sys.argv) > 2 and sys.argv[2] == "--test":
            # 仅运行测试
            await client.run_all_tests()
        else:
            # 先进行基本测试，然后进入交互模式
            print("\n正在运行基本测试...")
            await client.test_api_status()
            print("\n进入交互模式，你可以手动测试Tavily功能")
            await client.interactive_mode()

    except Exception as e:
        print(f"出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
