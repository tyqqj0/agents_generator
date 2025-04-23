# MCP 服务器

这个目录包含了多个MCP (Model Context Protocol) 服务器的实现，这些服务器可以被AI助手（如Claude）直接调用。

## 可用服务器

### 1. 天气服务器 (weather_mcp.py)

提供城市天气查询功能，包括：
- 获取指定城市的当前天气信息
- 列出所有可查询天气的城市
- 获取指定城市的未来天气预报（模拟数据）

### 2. 计算器服务器 (calculator_mcp.py)

提供基本的数学计算功能，包括：
- 加法、减法、乘法和除法
- 幂运算
- 平方根计算

### 3. Tavily 搜索服务器 (tavily_mcp.py)

提供互联网搜索和内容提取功能，包括：
- 网络搜索（一般搜索和新闻搜索）
- 网页内容提取
- API状态检查

## 如何使用

### Tavily MCP 服务器

要使用Tavily MCP服务器，需要先设置Tavily API密钥：

```bash
export TAVILY_API_KEY="你的Tavily API密钥"
```

然后启动服务器：

```bash
python agent_generator/mcp_servers/servers/tavily_mcp.py
```

### 示例用法

#### 1. 使用Tavily搜索

```python
result = await tavily_search("人工智能的最新发展", search_depth="advanced", max_results=3)
```

#### 2. 使用Tavily新闻搜索

```python
news = await tavily_news_search("中国科技新闻", days=3, max_results=5)
```

#### 3. 使用Tavily提取网页内容

```python
content = await tavily_extract("https://example.com", include_images=True)
```

## 配置说明

### Tavily MCP 服务器配置

Tavily MCP服务器支持以下主要功能：

1. **tavily_search**: 使用Tavily API执行网络搜索
   - 支持基础和高级搜索深度
   - 支持常规和新闻主题搜索
   - 可以过滤特定域名
   - 支持包含生成的答案、图片和原始内容

2. **tavily_news_search**: 专门用于新闻搜索的简化接口
   - 可以指定查询多少天之内的新闻
   - 提供与一般搜索相同的过滤选项

3. **tavily_extract**: 从指定URL提取网页内容
   - 支持提取单个或多个URL（最多20个）
   - 可以选择包含图片
   - 支持基础和高级提取深度

每个工具都有详细的文档，包括参数说明和使用示例。

## 添加新的MCP服务器

如果想添加新的MCP服务器，可以参考现有的实现，创建新的Python文件，使用`FastMCP`库实现所需功能。 