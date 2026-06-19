"""FastAPI command-center server: the whole fleet live, with overlays + controls. [v2 Phase E]

Thin HTTP layer over :class:`FleetSimManager`: serves the single-page command center, a JSON telemetry
feed (per-robot phase/target, per-machine true+perceived state, assignments, live accuracy), an MJPEG
top-down schematic with planned paths overlaid, and two controls — the controller toggle and the
perception-difficulty slider. ``make dashboard`` runs ``uvicorn feathersim.dashboard.fleet_server:app``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from feathersim.dashboard.fleet_manager import FleetSimManager
from feathersim.dashboard.server import _MJPEG_BOUNDARY, _mjpeg

_INDEX_HTML = Path(__file__).resolve().parent / "static" / "fleet.html"


class ControllerCommand(BaseModel):
    name: str


class DifficultyCommand(BaseModel):
    value: float


def create_app(manager: FleetSimManager | None = None) -> FastAPI:
    """Build the command-center app. If ``manager`` is None, a real one is created on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mgr = manager if manager is not None else FleetSimManager()
        app.state.manager = mgr
        mgr.start()
        try:
            yield
        finally:
            mgr.stop()

    app = FastAPI(title="FeatherSim Command Center", lifespan=lifespan)

    def _manager(request: Request) -> FleetSimManager:
        return request.app.state.manager

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _INDEX_HTML.read_text()

    @app.get("/api/telemetry")
    async def telemetry(request: Request) -> JSONResponse:
        return JSONResponse(_manager(request).telemetry())

    @app.post("/api/controller")
    async def controller(cmd: ControllerCommand, request: Request) -> JSONResponse:
        mgr = _manager(request)
        try:
            mgr.set_controller(cmd.name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse({"controller": mgr.controller_name})

    @app.post("/api/difficulty")
    async def difficulty(cmd: DifficultyCommand, request: Request) -> JSONResponse:
        mgr = _manager(request)
        mgr.set_difficulty(cmd.value)
        return JSONResponse({"difficulty": mgr.difficulty})

    @app.get("/api/camera")
    async def camera(request: Request) -> StreamingResponse:
        mgr = _manager(request)
        if mgr.frame() is None:
            raise HTTPException(status_code=503, detail="schematic not ready")
        return StreamingResponse(
            _mjpeg(mgr), media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}"
        )

    @app.get("/api/camera3d")
    async def camera3d(request: Request) -> StreamingResponse:
        mgr = _manager(request)
        if mgr.frame3d() is None:
            raise HTTPException(status_code=503, detail="3D feed unavailable (rendering disabled)")
        return StreamingResponse(
            _mjpeg(mgr, frame_getter=mgr.frame3d),
            media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
        )

    return app


app = create_app()
