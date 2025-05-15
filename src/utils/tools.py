import json
from typing import Dict

from anthropic.types import OverloadedError
from litellm import APIError
from mcp import Tool

from .response_item import ResponseItem


def get_selected_model(model, platform):
    selected_model = "anthropic/claude-3-5-sonnet-20241022"  # 默认模型
    if platform and model:
        # 根据平台选择正确的模型格式
        if platform == "anthropic":
            selected_model = f"anthropic/{model}"
        elif platform == "openai":
            selected_model = model
        elif platform == "google":
            selected_model = f"google/{model}"
        elif platform == "mistral":
            selected_model = f"mistral/{model}"
    return selected_model


def prepare_tools(mcp_composer, platform="anthropic"):
    available_tools = []
    for tool in mcp_composer.tools_map.values():
        tool_obj = tool.to_new_name_tool()
        clean_tool = format_tool_for_platform(tool_obj, platform)
        available_tools.append(clean_tool)
    return available_tools

def format_tool_for_platform(tool_obj: Tool, platform: str) -> Dict:
    """根据不同平台格式化工具

    Args:
        tool_obj: 工具对象
        platform: 平台类型，如"anthropic"或"openai"

    Returns:
        适用于指定平台的工具格式
    """
    if platform == "anthropic":
        # Anthropic格式
        clean_tool = {
            "name": tool_obj.name,
            "description": tool_obj.description,
            "input_schema": tool_obj.inputSchema
        }
        # 移除可能导致错误的意外字段
        if isinstance(clean_tool["input_schema"], dict) and "custom" in clean_tool["input_schema"]:
            if "annotations" in clean_tool["input_schema"]["custom"]:
                del clean_tool["input_schema"]["custom"]["annotations"]
        return clean_tool
    if platform == "openai":
        # OpenAI格式
        clean_tool = {
            "type": "function",
            "function": {
                "name": tool_obj.name,
                "description": tool_obj.description,
                "parameters": tool_obj.inputSchema
            }
        }
        # 移除可能导致错误的意外字段
        if isinstance(clean_tool["function"]["parameters"], dict) and "custom" in clean_tool["function"]["parameters"]:
            if "annotations" in clean_tool["function"]["parameters"]["custom"]:
                del clean_tool["function"]["parameters"]["custom"]["annotations"]
        return clean_tool

    # 其他平台的默认格式
    clean_tool = {
        "name": tool_obj.name,
        "description": tool_obj.description,
        "parameters": tool_obj.inputSchema
    }
    # 移除可能导致错误的意外字段
    if isinstance(clean_tool["parameters"], dict) and "custom" in clean_tool["parameters"]:
        if "annotations" in clean_tool["parameters"]["custom"]:
            del clean_tool["parameters"]["custom"]["annotations"]

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
                tool_call_response = json.loads(item.text)
                if tool_call_response['type'] == 'image':
                    tool_results.append({
                        "type": "image",
                        "content": tool_call_response['content']
                    })
                    continue
                elif tool_call_response['type'] == 'audio':
                    tool_results.append({
                        "type": "audio",
                        "content": tool_call_response['content']
                    })
                    continue
                # 默认处理为文本
                if hasattr(item, 'text'):
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
