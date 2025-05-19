import os
from typing import Optional, AsyncGenerator, List, Dict, Any

from anthropic._exceptions import OverloadedError, RateLimitError, APIError
from dotenv import load_dotenv

from api.downstream_controller import DownstreamController
from utils.message_handler import prepare_messages
from utils.model_client import call_model
from utils.response_item import ResponseItem
from utils.tool_handler import format_tool_result, prepare_tool_call_message, prepare_tool_response_message
from utils.tools import call_tool, handle_api_error, prepare_tools, get_selected_model

load_dotenv()  # 加载 .env 文件
os.environ.get("ANTHROPIC_API_KEY")
os.environ.get("OPENAI_API_KEY")


class MCPClient:
    def __init__(self, mcp_composer: DownstreamController):
        self.mcp_composer = mcp_composer

    async def process_query_stream(self, query: str,
                                   platform: Optional[str] = None,
                                   model: Optional[str] = None,
                                   chat_history: Optional[list] = None) -> AsyncGenerator[ResponseItem, None]:
        """处理查询并以流的形式逐个返回响应项"""
        # 初始化参数
        platform = platform or "anthropic"
        selected_model = get_selected_model(model, platform)
        available_tools = prepare_tools(self.mcp_composer, platform)
        
        # 准备消息列表
        messages = await prepare_messages(query, chat_history, selected_model)
        
        try:
            # 初始调用模型
            response = await call_model(selected_model, messages, available_tools)
            
            # 处理初始响应
            if not response.choices or len(response.choices) == 0:
                return
                
            message = response.choices[0].message
            
            # 处理文本内容
            if message.content:
                yield ResponseItem(type="text", content=message.content)
            
            # 处理工具调用
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    async for item in self._handle_tool_call(tool_call, messages, selected_model, available_tools):
                        yield item
                    
        except Exception as error:
            # 处理错误
            error_response, _ = await handle_api_error(error, 0)
            if error_response:
                yield error_response
    
    async def _handle_tool_call(self, tool_call, messages: List[Dict[str, Any]], model: str, available_tools: List[Dict[str, Any]]):
        """处理单个工具调用，返回消息列表供后续处理"""
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        # 通知前端开始工具调用
        yield ResponseItem(
            type="tool_call_start",
            content=tool_name,
            tool_args=tool_args
        )
        
        # 调用工具并获取结果
        tool_results, result = await call_tool(tool_name, tool_args, self.mcp_composer)
        
        # 返回工具调用结果
        yield ResponseItem(
            type="tool_call",
            content=tool_name,
            tool_results=tool_results,
            tool_args=tool_args
        )
        
        # 更新消息列表
        messages.append(await prepare_tool_call_message(tool_call, tool_args))
        
        # 处理工具结果内容
        tool_result_content = await format_tool_result(result)
        
        # 添加工具响应
        messages.append(await prepare_tool_response_message(tool_call.id, tool_result_content))
        
        # 处理模型的后续响应
        try:
            response = await call_model(model, messages, available_tools)
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if not content:
                    yield ResponseItem(type="text", content="模型没有返回内容")
                else:
                    yield ResponseItem(type="text", content=content)
                
                # 处理媒体内容
                for item in tool_results:
                    if item["type"] in ["image", "audio"]:
                        yield ResponseItem(type=item['type'], content=item['content'])
                
                # 检查是否有后续工具调用
                if response.choices[0].message.tool_calls:
                    for next_tool_call in response.choices[0].message.tool_calls:
                        async for item in self._handle_tool_call(next_tool_call, messages, model, available_tools):
                            yield item
                        
        except (OverloadedError, RateLimitError, APIError) as api_error:
            yield ResponseItem(type="text", content=f"模型响应失败: {str(api_error)}")