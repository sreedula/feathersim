"""FastAPI teleop + telemetry dashboard server. [Phase 6]

Thin HTTP layer over :class:`SimManager`: it serves the single-page UI, a JSON telemetry feed, an MJPEG
camera stream, and two control endpoints (teleop seizes manual control; mode resumes autonomy). All sim
work lives on the manager's background thread — handlers only read its published snapshots or push
operator inputs, so nothing here touches MuJoCo directly.

``make dashboard`` runs ``uvicorn feathersim.dashboard.server:app`` (the module-level ``app`` builds a
real :class:`SimManager` on startup via the lifespan). Tests call :func:`create_app` with an injected,
headless manager.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from feathersim.dashboard.sim_manager import SimManager

_INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"
_MJPEG_BOUNDARY = "frame"


class TeleopCommand(BaseModel):
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


class ModeCommand(BaseModel):
    mode: str


def create_app(manager: SimManager | None = None) -> FastAPI:
    """Build the dashboard app. If ``manager`` is None, a real :class:`SimManager` is created on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mgr = manager if manager is not None else SimManager()
        app.state.manager = mgr
        mgr.start()
        try:
            yield
        finally:
            mgr.stop()

    app = FastAPI(title="FeatherSim Dashboard", lifespan=lifespan)

    def _manager(request: Request) -> SimManager:
        return request.app.state.manager

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _INDEX_HTML.read_text()

    @app.get("/api/telemetry")
    async def telemetry(request: Request) -> JSONResponse:
        return JSONResponse(_manager(request).telemetry())

    @app.post("/api/teleop")
    async def teleop(cmd: TeleopCommand, request: Request) -> JSONResponse:
        mgr = _manager(request)
        mgr.set_teleop(cmd.vx, cmd.vy, cmd.omega)  # seizes manual control
        return JSONResponse({"mode": mgr.mode})

    @app.post("/api/mode")
    async def mode(cmd: ModeCommand, request: Request) -> JSONResponse:
        mgr = _manager(request)
        try:
            mgr.set_mode(cmd.mode)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse({"mode": mgr.mode})

    @app.get("/api/camera")
    async def camera(request: Request) -> StreamingResponse:
        mgr = _manager(request)
        if mgr.frame() is None:
            raise HTTPException(status_code=503, detail="camera unavailable (rendering disabled)")
        return StreamingResponse(
            _mjpeg(mgr),
            media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
        )

    return app


def _mjpeg(manager: SimManager, *, fps: float = 20.0):
    """Yield the manager's latest JPEG as an MJPEG multipart stream (a sync generator Starlette
    iterates in a threadpool, so the ``sleep`` paces frames without blocking the event loop).

    Ends when the manager stops so shutdown can't leave a live generator spinning; Starlette also
    stops pulling it when the browser disconnects.
    """
    head = f"--{_MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\n\r\n".encode()
    period = 1.0 / fps
    while manager.is_running():
        frame = manager.frame()
        if frame is not None:
            yield head + frame + b"\r\n"
        time.sleep(period)


app = create_app()
