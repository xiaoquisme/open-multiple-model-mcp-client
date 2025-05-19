import json
from typing import Dict, Any

from mcp.types import TextContent


async def format_tool_result(result) -> str:
    """格式化工具调用结果"""
    tool_result_content = ""
    if not result.content:
        return tool_result_content
        
    if isinstance(result.content, list):
        for item in result.content:
            try:
                if isinstance(item, TextContent):
                    tool_result_response = {
                        "type": "text",
                        "content": item.text
                    }
                else:
                    tool_result_response = json.loads(item.text)
            except Exception:
                tool_result_response = {
                    "type": "text",
                    "content": item.text
                }
            if tool_result_response.get('type') in ('image', 'audio'):
                tool_result_response.pop('content', None)
                item.text = json.dumps(tool_result_response)
            tool_result_content += item.text + "\n"
    elif isinstance(result.content, str):
        tool_result_content = result.content
    else:
        tool_result_content = json.dumps(result.content)
        
    return tool_result_content

async def prepare_tool_call_message(tool_call, tool_args: str) -> Dict[str, Any]:
    """准备工具调用消息"""
    return {
        "role": "assistant", 
        "content": None, 
        "tool_calls": [{
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_args if isinstance(tool_args, str) else json.dumps(tool_args)
            }
        }]
    }

async def prepare_tool_response_message(tool_call_id: str, content: str) -> Dict[str, Any]:
    """准备工具响应消息"""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content.strip()
    }