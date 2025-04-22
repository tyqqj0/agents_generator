# 自定义MCP服务器

这个目录包含了自定义的Model Context Protocol (MCP) 服务器实现。MCP是一个开放协议，用于让语言模型应用与外部数据源和工具无缝集成。



## 开发自定义MCP服务器

要创建自己的MCP服务器，可以参考现有的实现，主要步骤如下：

1. 引入MCP服务器库：
   ```python
   from mcp.server.fastmcp import FastMCP
   ```

2. 初始化服务器：
   ```python
   mcp = FastMCP("my_server_name")
   ```

3. 使用装饰器定义工具：
   ```python
   @mcp.tool(name="tool_name", description="工具描述")
   async def my_tool(param1: str, param2: int) -> dict:
       # 工具实现逻辑
       return {"result": "处理结果"}
   ```

4. 启动服务器：
   ```python
   if __name__ == "__main__":
       mcp.run(transport="stdio")
   ```

## 示例

更新后的`main.py`文件包含了如何在代理中使用这些自定义MCP服务器的示例代码。运行`main.py`并选择相应的测试模式即可查看不同代理的工作方式。 