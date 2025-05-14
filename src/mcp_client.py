import json
import os
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

from anthropic._exceptions import OverloadedError, RateLimitError, APIError
from dotenv import load_dotenv
from litellm import completion
from mcp import Tool

from api.downstream_controller import DownstreamController

load_dotenv()  # 加载 .env 文件
os.environ.get("ANTHROPIC_API_KEY")
os.environ.get("OPENAI_API_KEY")


# 定义响应项数据结构（与main.py中的ChatResponseItem保持一致）
class ResponseItem:
    def __init__(self, type: str, content: str, alt_text: Optional[str] = None, tool_results: Optional[List[Dict[str, Any]]] = None):
        self.type = type
        self.content = content
        self.alt_text = alt_text
        self.tool_results = tool_results  # 用于存储工具调用的结果列表

class MCPClient:
    def __init__(self, mcp_composer: DownstreamController):
        self.mcp_composer = mcp_composer
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 2  # 重试间隔（秒）

    # 抽取共同的工具准备逻辑
    def _prepare_tools(self, platform="anthropic"):
        """准备可用工具列表
        
        Args:
            platform: 平台类型，如"anthropic"或"openai"
            
        Returns:
            适用于指定平台的工具列表
        """
        available_tools = []
        for tool in self.mcp_composer.tools_map.values():
            tool_obj = tool.to_new_name_tool()
            clean_tool = self._format_tool_for_platform(tool_obj, platform)
            available_tools.append(clean_tool)
        return available_tools

    @staticmethod
    def _format_tool_for_platform(tool_obj: Tool, platform: str) -> Dict:
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
    
    # 抽取工具调用逻辑
    async def _call_tool(self, tool_name, tool_args):
        """调用工具并处理结果"""
        tool_results = []
        result = None  # 确保result总是被初始化
        
        try:
            real_tool_name = self.mcp_composer.get_tool_by_control_name(tool_name).tool.name
            result = await (self.mcp_composer.get_server_by_tool_name(tool_name)
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
    
    async def process_query_stream(self, query: str, platform: Optional[str] = None, model: Optional[str] = None) -> AsyncGenerator[ResponseItem, None]:
        """处理查询并以流的形式逐个返回响应项"""
        messages = [{"role": "user", "content": query}]
        
        # 默认使用anthropic作为平台
        platform = platform or "anthropic"
        available_tools = self._prepare_tools(platform)
        
        # 根据选择的平台和模型确定要使用的模型
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
        
        # 错误处理和重试逻辑
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                # 调用API，使用选择的模型
                response = completion(
                    model=selected_model,
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                # Process the response based on the structure provided
                assistant_message_content = []
                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message
                    
                    # Handle text content
                    if message.content:
                        text_item = ResponseItem(type="text", content=message.content)
                        yield text_item
                        assistant_message_content.append({"type": "text", "text": message.content})
                    
                    # Handle tool calls if present
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = tool_call.function.arguments
                            
                            # Notify frontend about tool call
                            yield ResponseItem(
                                type="tool_call_start",
                                content=tool_name
                            )
                            
                            # Call the tool and get results
                            tool_results, result = await self._call_tool(tool_name, tool_args)
                            
                            # Return tool call results
                            yield ResponseItem(
                                type="tool_call",
                                content=tool_name,
                                tool_results=tool_results
                            )
                            
                            # 继续与模型对话
                            assistant_message_content = []
                            assistant_message_content.append({
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_args if isinstance(tool_args, str) else json.dumps(tool_args)
                                }
                            })
                            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_args if isinstance(tool_args, str) else json.dumps(tool_args)
                                }
                            }]})
                            
                            # 将工具结果转换为OpenAI格式
                            # 处理结果内容为字符串
                            tool_result_content = ""
                            if result.content:
                                if isinstance(result.content, list):
                                    for item in result.content:
                                        tool_result_response = json.loads(item.text)
                                        if tool_result_response['type'] in ['image', 'audio']:
                                            del tool_result_response['content']
                                            item.text = json.dumps(tool_result_response)
                                        tool_result_content += item.text + "\n"
                                elif isinstance(result.content, str):
                                    tool_result_content = result.content
                                else:
                                    tool_result_content = json.dumps(result.content)
                            
                            # 使用OpenAI格式的工具响应
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result_content.strip()
                            })
                            
                            # 处理模型的响应
                            try:
                                response = completion(
                                    model=selected_model,
                                    max_tokens=1000,
                                    messages=messages,
                                    tools=available_tools
                                )
                                # Handle the model's response after the tool call
                                if response.choices and len(response.choices) > 0:
                                    if not response.choices[0].message.content:
                                        yield ResponseItem(type="text", content="模型没有返回内容")
                                    yield ResponseItem(type="text", content=response.choices[0].message.content)
                                    for item in tool_results:
                                        if item["type"] in ["image", "audio"]:
                                            yield ResponseItem(type=item['type'], content=item['content'])
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
