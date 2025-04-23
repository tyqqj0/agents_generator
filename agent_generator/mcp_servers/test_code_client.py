# -*- coding: utf-8 -*-
"""
@File    :   test_code_client.py
@Time    :   2025/04/24 10:35:12
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   测试代码执行MCP服务器的客户端
"""

import asyncio
import sys
import os
import json
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


import dotenv
dotenv.load_dotenv()

# 判断是否导入了E2B的api_key
if os.environ.get("E2B_API_KEY"):
    print("导入了E2B的api_key")
else:
    print("未导入了E2B的api_key")
    
    
if os.environ.get("WARNING_MODE") == "off":
    import warnings
    warnings.filterwarnings("ignore")
    # 禁用OpenAI的HTTP请求日志
os.environ["OPENAI_LOG"] = "none"
# 设置MCP和其他库的日志级别为WARNING，减少INFO日志的输出
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
else:
    print("正常模式")


class CodeTestClient:
    def __init__(self):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None  # 用于与 MCP 服务器通信的会话
        self.exit_stack = AsyncExitStack()  # 用于管理资源
        self.tools = []  # 存储可用工具列表

    async def connect_to_server(self, server_script_path: str):
        """连接到代码执行 MCP 服务器

        Args:
            server_script_path: 服务器脚本的路径
        """
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

    async def test_python_code_execution(self):
        """测试Python代码执行功能"""
        if not self.session:
            print("错误: 未连接到服务器")
            return False

        test_cases = [
            {
                "description": "简单算术计算",
                "code": "print(2 + 2)",
                "expected_contains": "4",
            },
            {
                "description": "变量定义和使用",
                "code": "x = 10\ny = 20\nprint(x + y)",
                "expected_contains": "30",
            },
            {
                "description": "循环",
                "code": "sum = 0\nfor i in range(5):\n    sum += i\nprint(sum)",
                "expected_contains": "10",
            },
            {
                "description": "条件语句",
                "code": "x = 15\nif x > 10:\n    print('大于10')\nelse:\n    print('不大于10')",
                "expected_contains": "大于10",
            },
            {
                "description": "函数定义和调用",
                "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(5))",
                "expected_contains": "120",
            },
            {
                "description": "错误代码测试",
                "code": "print(undefined_variable)",
                "expected_contains": "NameError",
            },
        ]

        success = True
        print("\n开始测试Python代码执行功能:")

        for i, test in enumerate(test_cases):
            try:
                print(f"\n测试 {i+1}: {test['description']}")
                print(f"代码:\n{test['code']}")

                result = await self.session.call_tool("run_code", {"code": test["code"], "pure_str_output": True})

                # 获取结果内容
                content = result.content

                # 如果是列表，获取第一个元素
                if isinstance(content, list):
                    if len(content) > 0:
                        content = content[0]
                    else:
                        content = None

                # 如果是TextContent对象，获取text属性
                if hasattr(content, "text"):
                    content = content.text

                # 解析返回的JSON字符串
                if isinstance(content, str):
                    try:
                        result_json = json.loads(content)
                        stdout = result_json.get("stdout", "")
                        stderr = result_json.get("stderr", "")
                        print(f"标准输出: {stdout}")
                        if stderr:
                            print(f"标准错误: {stderr}")
                    except json.JSONDecodeError:
                        print(f"无法解析返回的JSON: {content}")
                        stdout = content
                        stderr = ""

                # 检查输出是否符合期望
                if test["expected_contains"] in stdout or test["expected_contains"] in stderr:
                    print(f"✓ 测试通过: 包含预期输出 '{test['expected_contains']}'")
                else:
                    print(f"✗ 测试失败: 未找到预期输出 '{test['expected_contains']}'")
                    success = False

            except Exception as e:
                print(f"✗ 测试出错: {str(e)}")
                import traceback

                traceback.print_exc()
                success = False

        print(f"\nPython代码执行测试{'全部通过' if success else '部分失败'}")
        return success

    # async def test_javascript_code_execution(self):
    #     """测试JavaScript代码执行功能"""
    #     if not self.session:
    #         print("错误: 未连接到服务器")
    #         return False

    #     # 检查服务器是否支持JavaScript执行
    #     tool_names = [tool.name for tool in self.tools]
    #     if "run_javascript" not in tool_names:
    #         print("服务器不支持JavaScript代码执行，跳过测试")
    #         return True

    #     test_cases = [
    #         {
    #             "description": "简单算术计算",
    #             "code": "console.log(2 + 2);",
    #             "expected_contains": "4",
    #         },
    #         {
    #             "description": "变量定义和使用",
    #             "code": "const x = 10;\nconst y = 20;\nconsole.log(x + y);",
    #             "expected_contains": "30",
    #         },
    #         {
    #             "description": "循环",
    #             "code": "let sum = 0;\nfor (let i = 0; i < 5; i++) {\n    sum += i;\n}\nconsole.log(sum);",
    #             "expected_contains": "10",
    #         },
    #         {
    #             "description": "条件语句",
    #             "code": "const x = 15;\nif (x > 10) {\n    console.log('大于10');\n} else {\n    console.log('不大于10');\n}",
    #             "expected_contains": "大于10",
    #         },
    #         {
    #             "description": "函数定义和调用",
    #             "code": "function factorial(n) {\n    if (n <= 1) {\n        return 1;\n    }\n    return n * factorial(n-1);\n}\n\nconsole.log(factorial(5));",
    #             "expected_contains": "120",
    #         },
    #         {
    #             "description": "错误代码测试",
    #             "code": "console.log(undefinedVariable);",
    #             "expected_contains": "undefined",
    #         },
    #     ]

    #     success = True
    #     print("\n开始测试JavaScript代码执行功能:")

    #     for i, test in enumerate(test_cases):
    #         try:
    #             print(f"\n测试 {i+1}: {test['description']}")
    #             print(f"代码:\n{test['code']}")

    #             result = await self.session.call_tool("run_javascript", {"code": test["code"], "pure_str_output": True})

    #             # 获取结果内容
    #             content = result.content

    #             # 如果是列表，获取第一个元素
    #             if isinstance(content, list):
    #                 if len(content) > 0:
    #                     content = content[0]
    #                 else:
    #                     content = None

    #             # 如果是TextContent对象，获取text属性
    #             if hasattr(content, "text"):
    #                 content = content.text

    #             # 解析返回的JSON字符串
    #             if isinstance(content, str):
    #                 try:
    #                     result_json = json.loads(content)
    #                     stdout = result_json.get("stdout", "")
    #                     stderr = result_json.get("stderr", "")
    #                     print(f"标准输出: {stdout}")
    #                     if stderr:
    #                         print(f"标准错误: {stderr}")
    #                 except json.JSONDecodeError:
    #                     print(f"无法解析返回的JSON: {content}")
    #                     stdout = content
    #                     stderr = ""

    #             # 检查输出是否符合期望
    #             if test["expected_contains"] in stdout or test["expected_contains"] in stderr:
    #                 print(f"✓ 测试通过: 包含预期输出 '{test['expected_contains']}'")
    #             else:
    #                 print(f"✗ 测试失败: 未找到预期输出 '{test['expected_contains']}'")
    #                 success = False

    #         except Exception as e:
    #             print(f"✗ 测试出错: {str(e)}")
    #             import traceback

    #             traceback.print_exc()
    #             success = False

    #     print(f"\nJavaScript代码执行测试{'全部通过' if success else '部分失败'}")
    #     return success

    async def interactive_mode(self):
        """进入交互模式，手动测试代码执行功能"""
        if not self.session:
            print("错误: 未连接到服务器")
            return

        tool_names = [tool.name for tool in self.tools]
        supports_js = "run_javascript" in tool_names

        print("\n进入交互模式。")
        print("可用命令:")
        print(" - python <代码>: 执行Python代码")
        if supports_js:
            print(" - javascript <代码> 或 js <代码>: 执行JavaScript代码")
        print(" - quit: 退出")

        while True:
            try:
                cmd = input("\n> ").strip()

                if cmd.lower() == "quit":
                    break

                if not cmd:
                    continue

                # 解析命令
                parts = cmd.split(maxsplit=1)
                if len(parts) < 2:
                    print("请输入代码")
                    continue

                command = parts[0].lower()
                code = parts[1]

                if command == "python":
                    result = await self.session.call_tool("run_code", {"code": code, "pure_str_output": True})
                elif (command == "javascript" or command == "js") and supports_js:
                    result = await self.session.call_tool("run_javascript", {"code": code, "pure_str_output": True})
                else:
                    print(f"未知命令: {command}")
                    continue

                # 获取结果内容
                content = result.content

                # 如果是列表，获取第一个元素
                if isinstance(content, list):
                    if len(content) > 0:
                        content = content[0]
                    else:
                        content = None

                # 如果是TextContent对象，获取text属性
                if hasattr(content, "text"):
                    content = content.text

                # 解析返回的JSON字符串
                if isinstance(content, str):
                    try:
                        result_json = json.loads(content)
                        stdout = result_json.get("stdout", "")
                        stderr = result_json.get("stderr", "")
                        
                        if stdout:
                            print("\n标准输出:")
                            print(stdout)
                        
                        if stderr:
                            print("\n标准错误:")
                            print(stderr)
                            
                    except json.JSONDecodeError:
                        print(f"无法解析返回的JSON: {content}")

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

    # 获取代码执行服务器脚本路径
    if len(sys.argv) > 1:
        server_path = sys.argv[1]
    else:
        # 默认路径
        server_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "servers", "code_mcp.py"
        )

    client = CodeTestClient()
    try:
        connected = await client.connect_to_server(server_path)
        if not connected:
            print("无法连接到服务器，退出")
            return

        # 运行Python代码测试
        await client.test_python_code_execution()

        # 运行JavaScript代码测试
        # await client.test_javascript_code_execution()

        # 进入交互模式
        await client.interactive_mode()

    except Exception as e:
        print(f"出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 