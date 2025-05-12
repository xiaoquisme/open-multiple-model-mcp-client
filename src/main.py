from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import json
import asyncio

import uvicorn
from fastapi import FastAPI
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
async def chat_stream_api(request: Request, chat_request: Optional[ChatRequest] = None, message: Optional[str] = None):
    """流式聊天API，使用SSE返回响应"""
    # 优先使用POST中的JSON数据，如果没有则使用URL查询参数
    query = None
    if chat_request:
        query = chat_request.message
    elif message:
        query = message
    else:
        return {"error": "缺少消息内容，请提供message参数"}
        
    async def event_generator():
        mcp_client: MCPClient = request.app.state.mcp_client
        
        # 使用新的流式处理方法
        async for item in mcp_client.process_query_stream(query):
            chat_item = ChatResponseItem(
                type=item.type,
                content=item.content,
                alt_text=item.alt_text if item.alt_text is not None else None,
                tool_results=item.tool_results
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
        tools = [tool for tool in server_kit.servers_tools_hierarchy_map.get(server_name, []) if server_kit.tools_enabled.get(tool)]
        result.append({"name": server_name, "tools": tools})
    return result

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=3333, reload=True)


if __name__ == "__main__":
    main()
