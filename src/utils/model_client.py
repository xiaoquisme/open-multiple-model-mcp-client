from typing import List, Dict, Any

from litellm import acompletion


async def call_model(model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Any:
    """调用模型API"""
    model = f"openrouter/{model}" if not model.startswith("openrouter") else model
    return await acompletion(
        model=model,
        max_tokens=1000,
        messages=messages,
        tools=tools
    )

async def filter_relevant_history(query: str, chat_history: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """筛选与当前问题相关的聊天历史
    
    Args:
        query: 当前用户问题
        chat_history: 完整聊天历史
        model: 使用的模型
        
    Returns:
        筛选后的相关聊天历史
    """
    # 如果历史消息少于5条，直接返回全部历史
    if len(chat_history) <= 5:
        return chat_history
    
    # 准备历史消息的简短摘要
    history_summary = []
    for i, msg in enumerate(chat_history):
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            # 截断过长的消息内容
            content = msg['content']
            if len(content) > 100:
                content = content[:97] + "..."
            history_summary.append(f"{i+1}. [{msg['role']}]: {content}")

    # 构建提示
    prompt = [
        {"role": "system", "content": "你是一个帮助筛选相关聊天历史的助手。请分析用户当前的问题，并从历史消息中选择与当前问题最相关的消息。只返回相关消息的编号，用逗号分隔。如果所有消息都相关，返回'all'。"},
        {"role": "user", "content": f"当前问题: {query}\n\n历史消息:\n" + "\n".join(history_summary) + "\n\n请返回与当前问题最相关的历史消息编号，用逗号分隔。如果所有消息都相关，返回'all'。"}
    ]

    try:
        # 调用模型进行判断
        response = await call_model(model, prompt, [])
        
        result = response.choices[0].message.content.strip()
        
        # 处理模型返回结果
        if result.lower() == 'all':
            return chat_history
        
        # 解析返回的消息编号
        relevant_indices = []
        for part in result.replace(' ', '').split(','):
            try:
                idx = int(part) - 1  # 转换为0-based索引
                if 0 <= idx < len(chat_history):
                    relevant_indices.append(idx)
            except ValueError:
                continue
            
        # 如果没有找到相关消息，返回最近的3条消息
        if not relevant_indices:
            return chat_history[-3:] if len(chat_history) > 3 else chat_history
        
        # 返回筛选后的历史消息，并保持原有顺序
        relevant_indices.sort()
        return [chat_history[i] for i in relevant_indices]
    
    except Exception as e:
        # 出错时返回最近的几条消息
        return chat_history[-5:] if len(chat_history) > 5 else chat_history