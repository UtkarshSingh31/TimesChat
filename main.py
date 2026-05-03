import json
import aiosqlite
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from state import ChatRequest
from engine import workflow
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

app_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_graph
    async with aiosqlite.connect("persistence.db") as conn:
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        app_graph = workflow.compile(checkpointer=checkpointer)
        yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_generator(message: str, thread_id: str):
    # This thread_id is the UUID sent from your frontend script
    config = {"configurable": {"thread_id": thread_id}}
    
    async for event in app_graph.astream_events(
        {"messages": [("user", message)]}, 
        config, 
        version="v2"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
        elif kind == "on_tool_start":
            yield f"data: {json.dumps({'type': 'tool', 'content': 'Searching AI news...'})}\n\n"

    yield "data: [DONE]\n\n"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        stream_generator(request.message, request.thread_id),
        media_type="text/event-stream"
    )


@app.get("/sessions")
async def get_sessions():
    async with aiosqlite.connect("persistence.db") as conn:
        cursor = await conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC"
        )
        rows = await cursor.fetchall()
        return [{"id": row[0], "name": f"Chat {row[0][:8]}"} for row in rows]

@app.delete("/sessions/{thread_id}")
async def delete_session(thread_id: str):
    async with aiosqlite.connect("persistence.db") as conn:
        await conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        await conn.commit()
    return {"status": "success", "message": f"Session {thread_id} deleted"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)