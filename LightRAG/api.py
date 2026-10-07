import json
import sys
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from LightRAG.deep_agent import ask

if sys.platform == "win32":
    import asyncio      
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

MODEL_ID = "konstitusiya-agent"


@app.get("/v1/models")
def list_models():
    """LibreChat 'fetch: true' ilə model siyahısını buradan öyrənə bilər."""
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Son user mesajını götürürük — deep_agent.py-dəki ask() bir sualı
    # tam agent pipeline-ından (RAG tool daxil) keçirir.
    user_messages = [m for m in messages if m.get("role") == "user"]
    question = user_messages[-1]["content"] if user_messages else ""
    if isinstance(question, list):  # bəzi client-lər content-i block kimi göndərir
        question = " ".join(
            b.get("text", "") for b in question if isinstance(b, dict)
        )

    answer = ask(question)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if not stream:
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    # LibreChat defolt olaraq stream=true göndərir — cavabı SSE formatında,
    # bir dəfəlik "chunk" kimi qaytarırıq (real token-by-token streaming
    # yoxdur, çünki deep_agent.py-nin ask() funksiyası tam cavabı bir dəfəyə qaytarır).
    def event_stream():
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": answer},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")