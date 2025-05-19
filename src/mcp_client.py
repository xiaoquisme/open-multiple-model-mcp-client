import json
import os
import time
from typing import Optional, AsyncGenerator

from anthropic._exceptions import OverloadedError, RateLimitError, APIError
from dotenv import load_dotenv
from litellm import acompletion

from api.downstream_controller import DownstreamController
from utils.response_item import ResponseItem
from utils.tools import format_tool_for_platform, call_tool, handle_api_error, prepare_tools, get_selected_model

load_dotenv()  # 加载 .env 文件
os.environ.get("ANTHROPIC_API_KEY")
os.environ.get("OPENAI_API_KEY")

class MCPClient:
    def __init__(self, mcp_composer: DownstreamController):
        self.mcp_composer = mcp_composer
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 2  # 重试间隔（秒）

    # 抽取共同的工具准备逻辑


    async def process_query_stream(self, query: str,
                                   platform: Optional[str] = None,
                                   model: Optional[str] = None,
                                   chat_history: Optional[list] = None) -> AsyncGenerator[ResponseItem, None]:
        """处理查询并以流的形式逐个返回响应项"""
        # 构建消息列表，包含聊天历史
        messages = []
        
        # 添加聊天历史（如果有）
        if chat_history and isinstance(chat_history, list):
            for msg in chat_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": query})
        
        # 默认使用anthropic作为平台
        platform = platform or "anthropic"
        available_tools = prepare_tools(self.mcp_composer, platform)
        selected_model = get_selected_model(model, platform)

        # 错误处理和重试逻辑
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                # 调用API，使用选择的模型
                response = await acompletion(
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
                                content=tool_name,
                                tool_args=tool_args
                            )
                            
                            # Call the tool and get results
                            tool_results, result = await call_tool(tool_name, tool_args, self.mcp_composer)
                            
                            # Return tool call results
                            yield ResponseItem(
                                type="tool_call",
                                content=tool_name,
                                tool_results=tool_results,
                                tool_args=tool_args
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
                                        if tool_result_response.get('type') in ('image', 'audio'):
                                            tool_result_response.pop('content', None)
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
                                response = await acompletion(
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
                error_response, should_break = await handle_api_error(error, retry_count)
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
