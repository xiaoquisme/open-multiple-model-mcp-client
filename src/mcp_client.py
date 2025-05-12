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
        available_tools = []
        for tool in self.mcp_composer.tools_map.values():
            tool_obj = tool.to_new_name_tool()
            # Create a clean tool definition that matches Anthropic's expected format
            clean_tool = {
                "name": tool_obj.name,
                "description": tool_obj.description,
                "input_schema": tool_obj.inputSchema
            }
            # Remove any unexpected fields that might be causing the error
            if isinstance(clean_tool["input_schema"], dict) and "custom" in clean_tool["input_schema"]:
                if "annotations" in clean_tool["input_schema"]["custom"]:
                    del clean_tool["input_schema"]["custom"]["annotations"]
            available_tools.append(clean_tool)

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
                        real_tool_name = self.mcp_composer.get_tool_by_control_name(tool_name).tool.name
                        
                        # 创建一个工具调用结果列表
                        tool_results = []
                        
                        try:
                            result = await (self.mcp_composer.get_server_by_tool_name(tool_name)
                                          .session.call_tool(real_tool_name, tool_args))
                            
                            # 处理工具返回的结果
                            # 检查工具结果是否包含图像或音频内容
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
                        
                        # 添加工具调用及其结果作为一个组合项
                        response_items.append(ResponseItem(
                            type="tool_call",
                            content=tool_name,  # 工具名称
                            tool_results=tool_results  # 工具结果列表
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
                
            except OverloadedError:
                # API过载错误
                retry_count += 1
                if retry_count > self.max_retries:
                    response_items.append(ResponseItem(
                        type="text", 
                        content="抱歉，Claude服务暂时过载，请稍后再试。"
                    ))
                else:
                    # 等待后重试
                    time.sleep(self.retry_delay)
                    
            except RateLimitError:
                # 速率限制错误
                response_items.append(ResponseItem(
                    type="text", 
                    content="抱歉，已达到API调用限制，请稍后再试。"
                ))
                break
                
            except APIError as e:
                # 其他API错误
                response_items.append(ResponseItem(
                    type="text", 
                    content=f"API错误: {str(e)}"
                ))
                break
                
            except Exception as e:
                # 未预期的错误
                response_items.append(ResponseItem(
                    type="text", 
                    content=f"处理请求时发生错误: {str(e)}"
                ))
                break
        
        # 如果没有任何响应内容，添加一个默认错误消息
        if not response_items:
            response_items.append(ResponseItem(
                type="text", 
                content="抱歉，无法处理您的请求，请稍后再试。"
            ))
            
        return response_items
    
    async def process_query_stream(self, query: str) -> AsyncGenerator[ResponseItem, None]:
        """处理查询并以流的形式逐个返回响应项"""
        messages = [{"role": "user", "content": query}]
        available_tools = []
        for tool in self.mcp_composer.tools_map.values():
            tool_obj = tool.to_new_name_tool()
            # Create a clean tool definition that matches Anthropic's expected format
            clean_tool = {
                "name": tool_obj.name,
                "description": tool_obj.description,
                "input_schema": tool_obj.inputSchema
            }
            # Remove any unexpected fields that might be causing the error
            if isinstance(clean_tool["input_schema"], dict) and "custom" in clean_tool["input_schema"]:
                if "annotations" in clean_tool["input_schema"]["custom"]:
                    del clean_tool["input_schema"]["custom"]["annotations"]
            available_tools.append(clean_tool)
        
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
                        text_item = ResponseItem(type="text", content=content.text)
                        yield text_item
                        assistant_message_content.append(content)
                    elif content.type == 'tool_use':
                        tool_name = content.name
                        tool_args = content.input
                        real_tool_name = self.mcp_composer.get_tool_by_control_name(tool_name).tool.name
                        
                        # 创建一个工具调用结果列表
                        tool_results = []
                        
                        try:
                            # 先发送工具调用开始的消息
                            yield ResponseItem(
                                type="tool_call_start",
                                content=tool_name
                            )
                            
                            result = await (self.mcp_composer.get_server_by_tool_name(tool_name)
                                          .session.call_tool(real_tool_name, tool_args))
                            
                            # 处理工具返回的结果
                            # 检查工具结果是否包含图像或音频内容
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
                        
                        # 添加工具调用及其结果作为一个组合项
                        yield ResponseItem(
                            type="tool_call",
                            content=tool_name,  # 工具名称
                            tool_results=tool_results  # 工具结果列表
                        )
                            
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
                            error_msg = f"模型响应失败: {str(api_error)}"
                            yield ResponseItem(type="text", content=error_msg)
                
                # 成功处理请求，跳出重试循环
                break
                
            except OverloadedError:
                # API过载错误
                retry_count += 1
                if retry_count > self.max_retries:
                    yield ResponseItem(
                        type="text", 
                        content="抱歉，Claude服务暂时过载，请稍后再试。"
                    )
                else:
                    # 等待后重试
                    time.sleep(self.retry_delay)
                    
            except RateLimitError:
                # 速率限制错误
                yield ResponseItem(
                    type="text", 
                    content="抱歉，已达到API调用限制，请稍后再试。"
                )
                break
                
            except APIError as e:
                # 其他API错误
                yield ResponseItem(
                    type="text", 
                    content=f"API错误: {str(e)}"
                )
                break
                
            except Exception as e:
                # 未预期的错误
                yield ResponseItem(
                    type="text", 
                    content=f"处理请求时发生错误: {str(e)}"
                )
                break
        
        # 如果没有任何响应内容，添加一个默认错误消息
        if retry_count > self.max_retries:
            yield ResponseItem(
                type="text", 
                content="抱歉，无法处理您的请求，请稍后再试。"
            )
