from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse

from api.composer import Composer
from config import Config
from api.downstream_controller import DownstreamController
from mcp_client import MCPClient

app = FastAPI()


class ChatRequest(BaseModel):
    message: str

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


@app.post("/chat", response_model=str)
async def chat_api(request: Request, chat_request: ChatRequest):
    mcp_client: MCPClient = request.app.state.mcp_client
    reply = await mcp_client.process_query(chat_request.message)
    return reply

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
