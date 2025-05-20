import json
from typing import Dict

from anthropic.types import OverloadedError
from litellm import APIError
from mcp import Tool
from mcp.types import TextContent

from .response_item import ResponseItem


def get_selected_model(model, platform):
    return f'{platform}/{model}'


def prepare_tools(mcp_composer, platform="anthropic"):
    available_tools = []
    for tool in mcp_composer.tools_map.values():
        tool_obj = tool.to_new_name_tool()
        clean_tool = format_tool_for_platform(tool_obj)
        available_tools.append(clean_tool)
    return available_tools

def format_tool_for_platform(tool_obj: Tool) -> Dict:
    clean_tool = {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": tool_obj.inputSchema
        }
    }
    return clean_tool

async def call_tool(tool_name, tool_args, mcp_composer):
    """调用工具并处理结果"""
    tool_results = []
    result = None  # 确保result总是被初始化

    try:
        real_tool_name = mcp_composer.get_tool_by_control_name(tool_name).tool.name
        result = await (mcp_composer.get_server_by_tool_name(tool_name)
                        .session.call_tool(real_tool_name, json.loads(tool_args)))

        # 处理工具返回的结果
        if result.content and isinstance(result.content, list):
            for item in result.content:
                if isinstance(item, TextContent):
                    tool_results.append({
                        "type": "text",
                        "content": item.text
                    })
                    continue
                try:
                    tool_call_response = json.loads(item.text)
                except Exception:
                    tool_call_response = {
                        "type": "text",
                        "content": item.text
                    }
                response_type = tool_call_response.get('type')
                if response_type in ['image', 'audio', 'text']:
                    tool_results.append({
                        "type": response_type,
                        "content": tool_call_response['content']
                    })
                elif hasattr(item, 'text'):
                    tool_results.append({
                        "type": "text",
                        "content": item.text
                    })
    except Exception as tool_error:
        tool_results.append({
            "type": "text",
            "content": f"工具调用失败: {str(tool_error)}"
        })
        # 当发生异常时创建一个空的结果对象
        result = type('EmptyResult', (), {'content': []})()

    return tool_results, result

async def handle_api_error(error, retry_count, max_retries=3):
    """处理API错误并生成适当的响应"""
    from litellm import RateLimitError
    if isinstance(error, OverloadedError):
        # API过载错误
        if retry_count >= max_retries:
            return ResponseItem(
                type="text",
                content="抱歉，Claude服务暂时过载，请稍后再试。"
            ), True
        return None, False
    elif isinstance(error, RateLimitError):
        # 速率限制错误
        return ResponseItem(
            type="text",
            content="抱歉，已达到API调用限制，请稍后再试。"
        ), True
    elif isinstance(error, APIError):
        # 其他API错误
        return ResponseItem(
            type="text",
            content=f"API错误: {str(error)}"
        ), True
    else:
        # 未预期的错误
        return ResponseItem(
            type="text",
            content=f"处理请求时发生错误: {str(error)}"
        ), True
