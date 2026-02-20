import json
from typing import Any

def openai_to_gemini(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages", [])
    contents = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            role = "model"
        elif role == "system":
            # Gemini system instructions are usually separate, but for simple proxying:
            role = "user" 
        
        contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })
    
    return {"contents": contents}

def openai_to_claude(payload: dict[str, Any]) -> dict[str, Any]:
    # Claude messages API is quite similar to OpenAI but has some differences
    new_payload = {
        "model": payload.get("model"),
        "messages": payload.get("messages", []),
        "max_tokens": payload.get("max_tokens", 4096),
        "stream": payload.get("stream", False)
    }
    # Filter out system message if present
    messages = []
    system = ""
    for msg in new_payload["messages"]:
        if msg.get("role") == "system":
            system = msg.get("content")
        else:
            messages.append(msg)
    
    new_payload["messages"] = messages
    if system:
        new_payload["system"] = system
        
    return new_payload
