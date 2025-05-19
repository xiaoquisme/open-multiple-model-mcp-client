from typing import List, Dict, Any, Optional

from .model_client import filter_relevant_history

async def prepare_messages(query: str, chat_history: Optional[List[Dict[str, Any]]], selected_model: str) -> List[Dict[str, Any]]:
    """准备消息列表，包括处理聊天历史"""
    messages = []
    
    # 添加聊天历史（如果有）
    if chat_history and isinstance(chat_history, list):
        relevant_history = await filter_relevant_history(query, chat_history, selected_model)
        for msg in relevant_history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
    
    # 添加当前用户消息
    messages.append({"role": "user", "content": query})
    return messages