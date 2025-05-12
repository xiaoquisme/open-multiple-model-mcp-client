import os
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

from anthropic import Anthropic
from anthropic._exceptions import OverloadedError, RateLimitError, APIError
from dotenv import load_dotenv

from api.downstream_controller import DownstreamController

load_dotenv()  # 加载 .env 文件

# 定义响应项数据结构（与main.py中的ChatResponseItem保持一致）
class ResponseItem:
    def __init__(self, type: str, content: str, alt_text: Optional[str] = None, tool_results: Optional[List[Dict[str, Any]]] = None):
        self.type = type
        self.content = content
        self.alt_text = alt_text
        self.tool_results = tool_results  # 用于存储工具调用的结果列表

class MCPClient:
    def __init__(self, mcp_composer: DownstreamController):
        self.anthropic = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.mcp_composer = mcp_composer
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 2  # 重试间隔（秒）

    async def process_query(self, query: str) -> List[ResponseItem]:
        """处理查询并返回响应项列表（非流式）"""
        messages = [{"role": "user", "content": query}]
        available_tools = await self._prepare_tools()

        response_items = []
        
        # 添加错误处理和重试逻辑
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )
                
                assistant_message_content = []
                for content in response.content:
                    if content.type == 'text':
                        response_items.append(ResponseItem(type="text", content=content.text))
                        assistant_message_content.append(content)
                    elif content.type == 'tool_use':
                        tool_name = content.name
                        tool_args = content.input
                        
                        # 调用工具并获取结果
                        tool_results, result = await self._call_tool(tool_name, tool_args)
                        
                        # 添加工具调用及其结果
                        response_items.append(ResponseItem(
                            type="tool_call",
                            content=tool_name,
                            tool_results=tool_results
                        ))
                            
                        assistant_message_content.append(content)
                        messages.append({"role": "assistant", "content": assistant_message_content})
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }]
                        })
                        
                        # 工具调用后继续处理模型响应
                        try:
                            response = self.anthropic.messages.create(
                                model="claude-3-5-sonnet-20241022",
                                max_tokens=1000,
                                messages=messages,
                                tools=available_tools
                            )
                            # 处理工具调用后模型的回复
                            if response.content and len(response.content) > 0:
                                if response.content[0].type == 'text':
                                    response_items.append(ResponseItem(type="text", content=response.content[0].text))
                        except (OverloadedError, RateLimitError, APIError) as api_error:
                            error_msg = f"模型响应失败: {str(api_error)}"
                            response_items.append(ResponseItem(type="text", content=error_msg))
                
                # 成功处理请求，跳出重试循环
                break
                
            except Exception as error:
                # 处理错误
                error_response, should_break = await self._handle_api_error(error, retry_count)
                if error_response:
                    response_items.append(error_response)
                if should_break:
                    break
                    
                # 如果是过载错误且未达最大重试次数，等待后重试
                if isinstance(error, OverloadedError) and retry_count < self.max_retries:
                    retry_count += 1
                    time.sleep(self.retry_delay)
                else:
                    # 其他错误，直接跳出
                    break
        
        # 如果没有任何响应内容，添加一个默认错误消息
        if not response_items:
            response_items.append(ResponseItem(
                type="text", 
                content="抱歉，无法处理您的请求，请稍后再试。"
            ))
            
        return response_items
    
    # 抽取共同的工具准备逻辑
    async def _prepare_tools(self):
        """准备可用工具列表"""
        available_tools = []
        for tool in self.mcp_composer.tools_map.values():
            tool_obj = tool.to_new_name_tool()
            # 创建干净的工具定义，符合Anthropic预期的格式
            clean_tool = {
                "name": tool_obj.name,
                "description": tool_obj.description,
                "input_schema": tool_obj.inputSchema
            }
            # 移除可能导致错误的意外字段
            if isinstance(clean_tool["input_schema"], dict) and "custom" in clean_tool["input_schema"]:
                if "annotations" in clean_tool["input_schema"]["custom"]:
                    del clean_tool["input_schema"]["custom"]["annotations"]
            available_tools.append(clean_tool)
        return available_tools
    
    # 抽取工具调用逻辑
    async def _call_tool(self, tool_name, tool_args):
        """调用工具并处理结果"""
        tool_results = []
        try:
            real_tool_name = self.mcp_composer.get_tool_by_control_name(tool_name).tool.name
            result = await (self.mcp_composer.get_server_by_tool_name(tool_name)
                          .session.call_tool(real_tool_name, tool_args))
            
            # 处理工具返回的结果
            if result.content and isinstance(result.content, list):
                for item in result.content:
                    if hasattr(item, 'type') and item.type == 'image':
                        if hasattr(item, 'source') and hasattr(item.source, 'url'):
                            tool_results.append({
                                "type": "image",
                                "content": item.source.url,
                                "alt_text": getattr(item, 'alt', None)
                            })
                            continue
                    elif hasattr(item, 'type') and item.type == 'audio':
                        if hasattr(item, 'source') and hasattr(item.source, 'url'):
                            tool_results.append({
                                "type": "audio",
                                "content": item.source.url
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
        
        return tool_results, result
    
    # 处理Claude API错误的辅助方法
    async def _handle_api_error(self, error, retry_count):
        """处理API错误并生成适当的响应"""
        if isinstance(error, OverloadedError):
            # API过载错误
            if retry_count >= self.max_retries:
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
    
    async def process_query_stream(self, query: str) -> AsyncGenerator[ResponseItem, None]:
        """处理查询并以流的形式逐个返回响应项"""
        messages = [{"role": "user", "content": query}]
        available_tools = await self._prepare_tools()
        
        # 错误处理和重试逻辑
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                # 调用Claude API
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )
                
                assistant_message_content = []
                for content in response.content:
                    if content.type == 'text':
                        # 文本响应
                        text_item = ResponseItem(type="text", content=content.text)
                        yield text_item
                        assistant_message_content.append(content)
                    elif content.type == 'tool_use':
                        # 工具调用
                        tool_name = content.name
                        tool_args = content.input
                        
                        # 通知前端工具调用开始
                        yield ResponseItem(
                            type="tool_call_start",
                            content=tool_name
                        )
                        
                        # 调用工具并获取结果
                        tool_results, result = await self._call_tool(tool_name, tool_args)
                        
                        # 返回工具调用结果
                        yield ResponseItem(
                            type="tool_call",
                            content=tool_name,
                            tool_results=tool_results
                        )
                            
                        # 继续与模型对话
                        assistant_message_content.append(content)
                        messages.append({"role": "assistant", "content": assistant_message_content})
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }]
                        })
                        
                        # 工具调用后继续处理模型响应
                        try:
                            response = self.anthropic.messages.create(
                                model="claude-3-5-sonnet-20241022",
                                max_tokens=1000,
                                messages=messages,
                                tools=available_tools
                            )
                            # 处理工具调用后模型的回复
                            if response.content and len(response.content) > 0:
                                if response.content[0].type == 'text':
                                    yield ResponseItem(type="text", content=response.content[0].text)
                        except (OverloadedError, RateLimitError, APIError) as api_error:
                            yield ResponseItem(type="text", content=f"模型响应失败: {str(api_error)}")
                
                # 成功处理请求，跳出重试循环
                break
                
            except Exception as error:
                # 处理错误
                error_response, should_break = await self._handle_api_error(error, retry_count)
                if error_response:
                    yield error_response
                if should_break:
                    break
                    
                # 如果是过载错误且未达最大重试次数，等待后重试
                if isinstance(error, OverloadedError) and retry_count < self.max_retries:
                    retry_count += 1
                    time.sleep(self.retry_delay)
                else:
                    # 其他错误，直接跳出
                    break
