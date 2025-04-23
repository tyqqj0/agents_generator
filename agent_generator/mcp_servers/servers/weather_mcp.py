"""
天气MCP服务器

一个使用Python实现的简单天气MCP服务器，提供城市天气查询功能。
"""

import random
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP server
mcp = FastMCP("weather")

# 服务器配置信息
SERVER_NAME = "weather"
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
    "description": "天气信息MCP服务器",
}

# 模拟天气数据库
weather_db = {
    "北京": {
        "temperature": 22,
        "condition": "晴朗",
        "humidity": 35,
        "wind": "东北风 3级",
    },
    "上海": {
        "temperature": 24,
        "condition": "多云",
        "humidity": 60,
        "wind": "东风 2级",
    },
    "广州": {"temperature": 28, "condition": "雨", "humidity": 85, "wind": "南风 4级"},
    "深圳": {
        "temperature": 27,
        "condition": "阵雨",
        "humidity": 80,
        "wind": "南风 3级",
    },
    "成都": {
        "temperature": 20,
        "condition": "多云",
        "humidity": 55,
        "wind": "西南风 2级",
    },
    "杭州": {
        "temperature": 23,
        "condition": "晴朗",
        "humidity": 50,
        "wind": "东南风 2级",
    },
    "武汉": {
        "temperature": 25,
        "condition": "晴朗",
        "humidity": 45,
        "wind": "北风 2级",
    },
    "西安": {
        "temperature": 19,
        "condition": "晴朗",
        "humidity": 40,
        "wind": "西北风 3级",
    },
    "重庆": {
        "temperature": 22,
        "condition": "多云",
        "humidity": 65,
        "wind": "东北风 1级",
    },
    "南京": {
        "temperature": 21,
        "condition": "晴朗",
        "humidity": 55,
        "wind": "东风 2级",
    },
}


@mcp.tool(name="get_current_weather", description="获取指定城市的当前天气信息")
async def get_current_weather(city: str) -> Dict[str, Any]:
    """获取指定城市的当前天气信息。

    Args:
        city: 要查询天气的城市名称

    Returns:
        包含城市天气信息的字典

    Raises:
        ValueError: 当城市不在数据库中时
    """
    # 在模拟数据库中查找城市
    if city not in weather_db:
        raise ValueError(f'没有找到城市"{city}"的天气信息')

    weather = weather_db[city]

    # 返回天气信息
    return {
        "city": city,
        "temperature": f"{weather['temperature']}°C",
        "condition": weather["condition"],
        "humidity": f"{weather['humidity']}%",
        "wind": weather["wind"],
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@mcp.tool(name="list_available_cities", description="列出所有可查询天气的城市")
async def list_available_cities() -> List[str]:
    """列出所有可查询天气的城市。

    Returns:
        可用城市列表
    """
    # 返回可用城市列表
    return list(weather_db.keys())


@mcp.tool(
    name="get_weather_forecast", description="获取指定城市的未来天气预报（模拟数据）"
)
async def get_weather_forecast(city: str, days: int) -> Dict[str, Any]:
    """获取指定城市的未来天气预报。

    Args:
        city: 要查询天气预报的城市名称
        days: 预报天数（1-3天）

    Returns:
        包含城市天气预报的字典

    Raises:
        ValueError: 当城市不在数据库中时或者days不在1-3范围内
    """
    # 检查days参数
    if days < 1 or days > 3:
        raise ValueError("预报天数必须在1到3天之间")

    # 在模拟数据库中查找城市
    if city not in weather_db:
        raise ValueError(f'没有找到城市"{city}"的天气信息')

    current_weather = weather_db[city]

    # 生成模拟的天气预报
    forecast = []
    conditions = ["晴朗", "多云", "阴", "小雨", "阵雨", "雷阵雨"]
    today = datetime.now()

    for i in range(1, days + 1):
        forecast_date = today + timedelta(days=i)

        # 随机生成天气变化
        temp_change = random.randint(-3, 3)  # -3到3度的变化
        condition = random.choice(conditions)
        humidity_change = random.randint(-7, 7)  # -7到7%的变化

        humidity = max(30, min(100, current_weather["humidity"] + humidity_change))

        forecast.append(
            {
                "date": forecast_date.strftime("%Y-%m-%d"),
                "temperature": f"{current_weather['temperature'] + temp_change}°C",
                "condition": condition,
                "humidity": f"{humidity}%",
            }
        )

    return {"city": city, "forecast": forecast}


if __name__ == "__main__":
    # 启动MCP服务器
    # print(f"启动MCP服务器: {SERVER_NAME}")
    mcp.run(transport="stdio")
