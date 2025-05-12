# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is an Open Multiple Model MCP (Model Control Protocol) Client that allows for interfacing with multiple language models through a FastAPI-based gateway. It's designed to:

1. Run as a proxy that communicates with Anthropic's Claude via API
2. Expose MCP tools from downstream servers to Claude
3. Provide a unified FastAPI interface for clients to interact with

## Setup and Configuration

### Environment Setup

```bash
# Create a .env file from the example
cp .env-example .env
# Add your Anthropic API key to the .env file
```

### Running the Application

```bash
# Start the application
python src/main.py
```

By default, the server runs on port 3333 and can be accessed at `http://localhost:3333`.

## Core Architecture

The application follows a layered architecture:

1. **API Layer** (`main.py`): FastAPI application that exposes chat endpoints
2. **Client Layer** (`mcp_client.py`): Handles communication with Anthropic's API
3. **Controller Layer** (`downstream_controller.py`): Manages downstream MCP servers
4. **Composer Layer** (`composer.py`): Orchestrates server kits and gateways
5. **Gateway Layer** (`gateway.py`): Provides MCP-compatible endpoints

### Key Components

- **DownstreamMCPServer**: Represents a server providing MCP tools
- **ServerKit**: Configuration object for enabled/disabled servers and tools
- **Gateway**: Exposes MCP endpoints for specific server kits
- **Composer**: Manages the composition of multiple server kits and gateways

## Configuration

The application is configured via:

1. Environment variables (.env file)
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - `HOST`: Host to bind the server to (default: 0.0.0.0)
   - `PORT`: Port to run the server on (default: 8000)
   - `MCP_COMPOSER_PROXY_URL`: URL for the MCP composer proxy (default: http://localhost:8000)
   - `MCP_SERVERS_CONFIG_PATH`: Path to the MCP servers configuration file (default: ./mcp_servers.json)

2. MCP Servers JSON Configuration
   - Defines the downstream MCP servers to connect to
   - Supports both stdio (command-based) and SSE (URL-based) connections

## Adding New MCP Servers

To add a new MCP server, update the `mcp_servers.json` file:

```json
{
  "mcpServers": {
    "server_name": {
      "command": "command_to_run",
      "args": ["arg1", "arg2"],
      "env": {}
    },
    "another_server": {
      "url": "http://server-url"
    }
  }
}
```

## API Endpoints

- **POST /chat**: Main chat endpoint
  - Accepts: `{ "message": "your message" }`
  - Returns: Text response from Claude, including tool use results

- **MCP Gateway**: `/mcp/{server_kit_name}`
  - `/mcp/{server_kit_name}/sse`: SSE endpoint
  - `/mcp/{server_kit_name}/messages`: Messages endpoint