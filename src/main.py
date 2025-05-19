from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import json
import asyncio
import os  # 添加os导入

import uvicorn
from fastapi import FastAPI, Query  # 添加Query导入
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from api.composer import Composer
from config import Config
from api.downstream_controller import DownstreamController
from mcp_client import MCPClient

app = FastAPI()


class ChatRequest(BaseModel):
    message: str

class ChatResponseItem(BaseModel):
    type: str  # "text", "audio", "image", "tool_call" 等
    content: str  # 内容、URL或Base64编码的数据
    alt_text: Optional[str] = None  # 可选的替代文本
    tool_results: Optional[List[Dict[str, Any]]] = None  # 工具调用结果列表
    tool_args: Optional[str] = None  # 工具调用参数

class ChatResponse(BaseModel):
    items: list[ChatResponseItem]
    conversation_id: str = None

# 请根据实际 MCP 服务地址修改 base_url
config = Config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI server start
    downstream_controller = DownstreamController(config.servers)
    await downstream_controller.initialize()

    # Initialize and store in app.state after controller is ready
    # Pass config to Composer
    composer = Composer(downstream_controller, config)
    app.state.composer = composer
    app.state.mcp_client = MCPClient(downstream_controller)
    server_kit = composer.create_server_kit("composer")
    await composer.add_gateway(server_kit)
    app.mount("/mcp/", app.state.composer.asgi_gateway_routes())

    yield
    # FastAPI server shutdown
    await downstream_controller.shutdown()


app = FastAPI(debug=True, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 新的流式聊天API
@app.post("/chat_stream")
@app.get("/chat_stream")  # 添加GET方法支持
async def chat_stream_api(
    request: Request, 
    chat_request: Optional[ChatRequest] = None, 
    message: Optional[str] = None,
    platform: Optional[str] = None,  # 新增平台参数
    model: Optional[str] = None,     # 新增模型参数
    chat_history: Optional[str] = None  # 新增聊天历史参数
):
    """流式聊天API，使用SSE返回响应"""
    # 优先使用POST中的JSON数据，如果没有则使用URL查询参数
    query = None
    if chat_request:
        query = chat_request.message
    elif message:
        query = message
    else:
        return {"error": "缺少消息内容，请提供message参数"}
        
    # 解析聊天历史
    history = []
    if chat_history:
        try:
            history = json.loads(chat_history)
        except json.JSONDecodeError:
            # 解析失败时使用空历史
            history = []
            
    async def event_generator():
        mcp_client: MCPClient = request.app.state.mcp_client
        
        # 使用新的流式处理方法
        # 传递平台、模型和聊天历史参数
        async for item in mcp_client.process_query_stream(query, platform, model, history):
            chat_item = ChatResponseItem(
                type=item.type,
                content=item.content,
                alt_text=item.alt_text if item.alt_text is not None else None,
                tool_results=item.tool_results,
                tool_args=item.tool_args if item.tool_args is not None else None
            )
            # 将响应项转换为JSON并发送
            yield {
                "event": "message",
                "data": chat_item.model_dump_json()
            }
            # 添加一个小延迟，以确保前端能够处理每个消息
            await asyncio.sleep(0.01)
        
        # 发送完成事件
        yield {
            "event": "done",
            "data": json.dumps({"status": "complete"})
        }
        
    return EventSourceResponse(event_generator())

@app.get("/")
async def read_root():
    return FileResponse("src/ui/index.html")

@app.get("/servers", response_model=list)
async def list_servers(request: Request):
    composer = request.app.state.composer
    server_kit = await composer.get_server_kit("composer")
    result = []
    for server_name in server_kit.servers_enabled.keys():
        tools_info = []
        for tool_name in server_kit.servers_tools_hierarchy_map.get(server_name, []):
            if server_kit.tools_enabled.get(tool_name):
                # 获取工具对象以提取描述信息
                tool_obj = composer.downstream_controller.get_tool_by_control_name(tool_name)
                tools_info.append({
                    "name": tool_name,
                    "description": tool_obj.tool.description if tool_obj and hasattr(tool_obj.tool, 'description') else "无描述信息"
                })
        result.append({"name": server_name, "tools": tools_info})
    return result

# 新增：获取可用平台列表
@app.get("/api/platforms")
async def get_platforms():
    # 提供支持的平台列表
    platforms = [
        {"id": "anthropic", "name": "Anthropic"},
        {"id": "openai", "name": "OpenAI"},
        {"id": "google", "name": "Google"},
        {"id": "mistral", "name": "Mistral"}
    ]
    # 如果存在环境变量，则根据环境变量筛选可用平台
    available_platforms = []
    for platform in platforms:
        if platform["id"] == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            available_platforms.append(platform)
        elif platform["id"] == "openai" and os.environ.get("OPENAI_API_KEY"):
            available_platforms.append(platform)
        elif platform["id"] == "google" and os.environ.get("GOOGLE_API_KEY"):
            available_platforms.append(platform)
        elif platform["id"] == "mistral" and os.environ.get("MISTRAL_API_KEY"):
            available_platforms.append(platform)
    
    # 如果没有可用平台（环境变量未设置），则返回所有平台
    if not available_platforms:
        return platforms
    
    return available_platforms

# 新增：根据平台获取模型列表
@app.get("/api/models")
async def get_models(platform: str = Query(..., description="平台ID，如anthropic, openai等")):
    # 各平台支持的模型列表
    models_by_platform = {
        "anthropic": [
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
            {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
            {"id": "claude-2.1", "name": "Claude 2.1"},
            {"id": "claude-2.0", "name": "Claude 2.0"}
        ],
        "openai": [
            {"id": "gpt-4o", "name": "GPT-4o"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "gpt-4", "name": "GPT-4"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
        ],
        "google": [
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
            {"id": "gemini-1.0-pro", "name": "Gemini 1.0 Pro"}
        ],
        "mistral": [
            {"id": "mistral-large-latest", "name": "Mistral Large"},
            {"id": "mistral-medium-latest", "name": "Mistral Medium"},
            {"id": "mistral-small-latest", "name": "Mistral Small"}
        ]
    }
    
    # 返回对应平台的模型列表，如果平台不存在则返回空列表
    return models_by_platform.get(platform, [])

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=3333, reload=True)


if __name__ == "__main__":
    main()
