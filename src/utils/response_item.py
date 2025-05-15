from typing import Optional, List, Dict, Any


class ResponseItem:
    def __init__(self, type: str, content: str, alt_text: Optional[str] = None, tool_results: Optional[List[Dict[str, Any]]] = None, tool_args: Optional[str] = None):
        self.type = type
        self.content = content
        self.alt_text = alt_text
        self.tool_results = tool_results  # 用于存储工具调用的结果列表
        self.tool_args = tool_args  # 用于存储工具调用的参数
