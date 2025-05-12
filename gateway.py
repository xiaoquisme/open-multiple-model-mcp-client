from server_kit import ServerKit
from downstream_server import DownstreamMCPServerTool, DownstreamMCPServer
from typing import Any
import logging  # Import logging
from mcp.types import (
    ServerResult,
    ListToolsResult,
    ListToolsRequest,
    CallToolRequest,
    CallToolResult,
    TextContent,
)
from mcp.server.lowlevel import Server
from downstream_controller import DownstreamController
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount

# Remove the import from .config

# Get logger for this module
logger = logging.getLogger(__name__)


class Gateway:
    def __init__(
        self,
        server_kit: ServerKit,
        downstream_controller: DownstreamController,
        mcp_composer_proxy_url: str,
    ):
        self.server_kit = server_kit
        self.downstream_controller = downstream_controller
        self.sse_path = f"/mcp/{self.server_kit.name}/sse"
        self.messages_path = f"/mcp/{self.server_kit.name}/messages"
        self.gateway_endpoint = f"{mcp_composer_proxy_url}{self.sse_path}"
        self.server = Server(name=self.server_kit.name)
        self.sse = SseServerTransport(self.messages_path)

    @property
    def name(self):
        return self.server_kit.name

    async def setup(self):
        async def _list_tools(_: Any) -> ServerResult:
            if not self.server_kit.enabled:
                return ServerResult(ListToolsResult(tools=[]))
            enabled_tool_names = self.server_kit.list_enabled_tool_names()
            tools = []
            for tool_name in enabled_tool_names:
                tool: DownstreamMCPServerTool = (
                    self.downstream_controller.get_tool_by_control_name(tool_name)
                )
                tools.append(tool)
            return ServerResult(
                ListToolsResult(tools=[t.to_new_name_tool() for t in tools])
            )

        self.server.request_handlers[ListToolsRequest] = _list_tools

        async def _call_tool(req: CallToolRequest) -> ServerResult:
            try:
                if not self.server_kit.enabled:
                    raise ValueError("Server kit is not enabled")
                tool_name = req.params.name
                tool_enabled = self.server_kit.tools_enabled[tool_name]
                if not tool_enabled:
                    raise ValueError("Tool is not enabled")
                tool: DownstreamMCPServerTool = (
                    self.downstream_controller.get_tool_by_control_name(tool_name)
                )
                server_control_name = tool.server_control_name
                server: DownstreamMCPServer = (
                    self.downstream_controller.get_server_by_control_name(
                        server_control_name
                    )
                )
                if server.session is None:
                    raise ValueError("Server session is not initialized")
                result = await server.session.call_tool(
                    tool.tool.name,
                    (req.params.arguments or {}),
                )
                return ServerResult(result)
            except Exception as e:  # noqa: BLE001
                return ServerResult(
                    CallToolResult(
                        content=[TextContent(type="text", text=str(e))],
                        isError=True,
                    ),
                )

        self.server.request_handlers[CallToolRequest] = _call_tool

    def as_asgi_route(self):
        async def handle_sse(request):
            async with self.sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self.server.run(
                    streams[0], streams[1], self.server.create_initialization_options()
                )

        return Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=self.sse.handle_post_message),
            ],
        )
