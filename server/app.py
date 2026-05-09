from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from speculate_forge.models import SpeculateAction, SpeculateConfig
from server.speculate_environment import SpeculateForgeEnvironment

ROOT_DIR = Path(__file__).resolve().parent.parent
UI_PATH = ROOT_DIR / "ui" / "index.html"


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        stale: list[WebSocket] = []
        async with self._lock:
            for connection in self._connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    stale.append(connection)
            for connection in stale:
                if connection in self._connections:
                    self._connections.remove(connection)


app = FastAPI(title="SpeculateForge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

environment = SpeculateForgeEnvironment()
connections = ConnectionManager()


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(UI_PATH)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "project": "SpeculateForge",
        "stage": "phase_e_multitask_foundation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/info")
async def info() -> dict:
    return environment.info()


@app.get("/schema")
async def schema() -> dict:
    return environment.schema()


@app.get("/metadata")
async def metadata() -> dict:
    return environment.metadata()


@app.get("/state")
async def state() -> dict:
    return environment.state().model_dump()


@app.post("/mcp")
async def mcp_stub() -> JSONResponse:
    return JSONResponse(
        {
            "status": "not_implemented",
            "message": "MCP integration is reserved for future OpenEnv compatibility work.",
        },
        status_code=501,
    )


@app.post("/reset")
async def reset(task_level: int = 1) -> dict:
    response = await asyncio.to_thread(environment.reset, task_level)
    await connections.broadcast(
        {
            "event": "reset",
            "task_level": task_level,
            "payload": response.model_dump(),
        }
    )
    return response.model_dump()


@app.post("/step")
async def step(action: SpeculateAction, task_level: int = 1) -> dict:
    response = await asyncio.to_thread(environment.step, action, task_level)
    await connections.broadcast(
        {
            "event": "step",
            "task_level": task_level,
            "payload": response.model_dump(),
        }
    )
    return response.model_dump()


@app.post("/manual_step")
async def manual_step(config: SpeculateConfig, task_level: int = 1) -> dict:
    response = await asyncio.to_thread(environment.manual_step, config, task_level)
    await connections.broadcast(
        {
            "event": "manual_step",
            "task_level": task_level,
            "payload": response,
        }
    )
    return response


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await connections.connect(websocket)
    try:
        await websocket.send_json(
            {
                "event": "connected",
                "payload": {
                    "message": "WebSocket connected",
                    "stage": "phase_e_multitask_foundation",
                },
            }
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await connections.disconnect(websocket)
