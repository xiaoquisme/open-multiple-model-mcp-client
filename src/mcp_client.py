import os

from anthropic import Anthropic
from dotenv import load_dotenv

from api.downstream_controller import DownstreamController

load_dotenv()  # 加载 .env 文件


class MCPClient:
    def __init__(self, mcp_composer: DownstreamController):
        self.anthropic = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.mcp_composer = mcp_composer

    async def process_query(self, query: str) -> str:
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

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        final_text = []
        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                real_tool_name = self.mcp_composer.get_tool_by_control_name(tool_name).tool.name
                result = await (self.mcp_composer.get_server_by_tool_name(tool_name)
                                .session.call_tool(real_tool_name, tool_args))
                final_text.append(f"[调用工具 {tool_name} 参数 {tool_args}]")
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
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )
                final_text.append(response.content[0].text)
        return "\n".join(final_text)
