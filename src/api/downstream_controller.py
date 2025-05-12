from api.downstream_server import (
    DownstreamMCPServerConfig,
    DownstreamMCPServer,
    DownstreamMCPServerTool,
)
from typing import Dict, List, Tuple
import asyncio
from contextlib import AsyncExitStack


class DownstreamController:
    def __init__(self, configs: List[DownstreamMCPServerConfig]):
        self._all_servers_tools: List[
            Tuple[DownstreamMCPServer, List[DownstreamMCPServerTool]]
        ] = []
        self._servers_map: Dict[str, DownstreamMCPServer] = {}
        self._tools_map: Dict[str, DownstreamMCPServerTool] = {}
        self._asyncio_lock = asyncio.Lock()
        self.configs = configs
        self.exit_stack = AsyncExitStack()
        self._initialized = False

    async def initialize(self):
        async with self._asyncio_lock:
            for config in self.configs:
                await self.register_downstream_mcp_server(config)
            self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self):
        async with self._asyncio_lock:
            for server, _ in self._all_servers_tools:
                await server.shutdown()
            await self.exit_stack.aclose()

    async def register_downstream_mcp_server(self, config: DownstreamMCPServerConfig):
        server = DownstreamMCPServer(config)
        await server.initialize(self.exit_stack)
        self._servers_map[server.get_control_name()] = server
        tools = await server.list_tools()
        self._all_servers_tools.append((server, tools))
        for tool in tools:
            self._tools_map[tool.control_name] = tool

    def list_all_servers_tools(
        self,
    ) -> List[Tuple[DownstreamMCPServer, List[DownstreamMCPServerTool]]]:
        return self._all_servers_tools

    def get_tool_by_control_name(
        self, tool_control_name: str
    ) -> DownstreamMCPServerTool:
        return self._tools_map[tool_control_name]

    def get_server_by_control_name(
        self, server_control_name: str
    ) -> DownstreamMCPServer:
        return self._servers_map[server_control_name]
    
    def get_server_by_tool_name(
        self, tool_name: str
    ) -> DownstreamMCPServer | None:

        downstream_tool: DownstreamMCPServerTool | None = self._tools_map.get(tool_name)
        if downstream_tool:
            return self._servers_map.get(downstream_tool.server_control_name)

    @property
    def tools_map(self) -> Dict[str, DownstreamMCPServerTool]:
        return self._tools_map