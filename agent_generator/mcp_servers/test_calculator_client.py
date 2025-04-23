# -*- coding: utf-8 -*-
"""
@File    :   test_calculator_client.py
@Time    :   2025/04/23 14:35:12
@Author  :   tyqqj
@Version :   1.0
@Contact :   tyqqj0@163.com
@Desc    :   None
"""





import asyncio
import sys
import os
import json
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class CalculatorTestClient:
    def __init__(self):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None  # 用于与 MCP 服务器通信的会话
        self.exit_stack = AsyncExitStack()  # 用于管理资源
        self.tools = []  # 存储可用工具列表

    async def connect_to_server(self, server_script_path: str):
        """连接到计算器 MCP 服务器

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

    async def test_calculator_tools(self):
        """测试计算器的所有功能"""
        if not self.session:
            print("错误: 未连接到服务器")
            return False

        test_cases = [
            {"tool": "add", "args": {"a": 5, "b": 3}, "expected": 8},
            {"tool": "subtract", "args": {"a": 10, "b": 4}, "expected": 6},
            {"tool": "multiply", "args": {"a": 7, "b": 6}, "expected": 42},
            {"tool": "divide", "args": {"a": 20, "b": 4}, "expected": 5},
            {"tool": "power", "args": {"base": 2, "exponent": 3}, "expected": 8},
            {"tool": "sqrt", "args": {"number": 16}, "expected": 4},
        ]

        success = True
        print("\n开始测试计算器功能:")

        for test in test_cases:
            try:
                print(f"测试: {test['tool']} 参数: {json.dumps(test['args'])}")
                result = await self.session.call_tool(test["tool"], test["args"])

                # 处理可能的content类型
                content = result.content
                print(f"原始结果: {content}, 类型: {type(content)}")

                # 如果是列表，获取第一个元素
                if isinstance(content, list):
                    if len(content) > 0:
                        content = content[0]
                    else:
                        content = 0

                # 如果是TextContent对象，获取text属性
                if hasattr(content, "text"):
                    content = content.text

                # 确保是数字
                result_value = float(content)
                passed = abs(result_value - test["expected"]) < 0.0001

                if passed:
                    print(f"✓ {test['tool']}({test['args']}): {result_value}")
                else:
                    print(
                        f"✗ {test['tool']}({test['args']}): 预期 {test['expected']}, 得到 {result_value}"
                    )
                    success = False

            except Exception as e:
                print(f"✗ {test['tool']}({test['args']}): 出错 - {str(e)}")
                import traceback

                traceback.print_exc()
                success = False

        print(f"\n测试{'全部通过' if success else '部分失败'}")
        return success

    async def interactive_mode(self):
        """进入交互模式，手动测试计算器功能"""
        if not self.session:
            print("错误: 未连接到服务器")
            return

        print("\n进入交互模式。格式: <工具名> <参数1> <参数2> ...")
        print("示例: add 5 3")
        print("输入 'quit' 退出")

        while True:
            try:
                cmd = input("\n> ").strip()

                if cmd.lower() == "quit":
                    break

                parts = cmd.split()
                if len(parts) < 1:
                    print("请输入工具名称和参数")
                    continue

                tool_name = parts[0]

                # 查找工具定义
                tool_def = None
                for tool in self.tools:
                    if tool.name == tool_name:
                        tool_def = tool
                        break

                if not tool_def:
                    print(f"未知工具: {tool_name}")
                    continue

                # 解析参数
                args = {}
                param_names = list(tool_def.inputSchema.get("properties", {}).keys())

                if len(parts) - 1 != len(param_names):
                    print(
                        f"参数数量不匹配。{tool_name}需要参数: {', '.join(param_names)}"
                    )
                    continue

                for i, param_name in enumerate(param_names):
                    try:
                        args[param_name] = float(parts[i + 1])
                    except ValueError:
                        print(f"无法将 '{parts[i+1]}' 转换为数字")
                        continue

                # 调用工具
                result = await self.session.call_tool(tool_name, args)
                content = result.content
                print(f"原始结果: {content}, 类型: {type(content)}")

                # 如果是列表，获取第一个元素
                if isinstance(content, list):
                    if len(content) > 0:
                        content = content[0]
                    else:
                        content = "空列表"

                # 如果是TextContent对象，获取text属性
                if hasattr(content, "text"):
                    content = content.text

                print(f"结果: {content}")

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

    # 获取计算器服务器脚本路径
    if len(sys.argv) > 1:
        server_path = sys.argv[1]
    else:
        # 默认路径
        server_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "servers", "calculator_mcp.py"
        )

    client = CalculatorTestClient()
    try:
        connected = await client.connect_to_server(server_path)
        if not connected:
            print("无法连接到服务器，退出")
            return

        # 先运行自动测试
        await client.test_calculator_tools()

        # 然后进入交互模式
        await client.interactive_mode()

    except Exception as e:
        print(f"出现错误: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
