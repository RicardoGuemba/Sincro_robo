from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import PROJECT_ROOT
from .services.runtime import ApplicationRuntime


class SessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    planes: list[float]


class Decision(BaseModel):
    confirm: bool


class CampaignCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    square_size_mm: float = Field(gt=0)


class CampaignChecks(BaseModel):
    camera_fixed: bool = False
    lens_focus_locked: bool = False
    production_resolution: bool = False
    single_stapi_client: bool = False


class CaptureDecision(BaseModel):
    decision: str
    reason: str = ""


class DirectionConfirm(BaseModel):
    sense: str


class PlaneSelect(BaseModel):
    plan_z: float


def _campaign(view: dict[str, Any] | None) -> dict[str, Any]:
    if view is None:
        raise HTTPException(status_code=404, detail="Nenhuma campanha ativa")
    return view


def create_app(config: dict[str, Any]) -> FastAPI:
    Path(config["app"]["exports_dir"]).mkdir(parents=True, exist_ok=True)
    runtime = ApplicationRuntime(config, PROJECT_ROOT)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(
        title="SINCRO_ROBO",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    web_dir = Path(__file__).resolve().parent / "web"
    app.mount("/assets", StaticFiles(directory=web_dir), name="assets")

    @app.exception_handler(ValueError)
    async def value_error_handler(_request, error: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        snapshot = runtime.state.snapshot()
        return {
            "status": "ok",
            "version": __version__,
            "hardware": snapshot["hardware"],
            "mode": {
                "camera": config["camera"]["provider"],
                "model": config["model"]["provider"],
                "plc": config["plc"]["provider"],
            },
        }

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return runtime.snapshot()

    @app.get("/api/frame")
    def frame() -> Response:
        with runtime.state._lock:
            jpeg = runtime.state.frame_jpeg
        if jpeg is None:
            raise HTTPException(status_code=503, detail="Frame ainda indisponível")
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/config")
    def public_config() -> dict[str, Any]:
        return {
            "version": __version__,
            "default_planes_mm": config["calibration"]["default_planes_mm"],
            "xy_tolerance_mm": config["calibration"]["xy_tolerance_mm"],
            "angular_tolerance_deg": config["calibration"]["angular_tolerance_deg"],
            "pick_offset_local_mm": config["calibration"]["pick_offset_local_mm"],
            "model_threshold": config["model"]["threshold"],
            "mode": {
                "camera": config["camera"]["provider"],
                "model": config["model"]["provider"],
                "plc": config["plc"]["provider"],
            },
            "model_hash": runtime.controller.model_hash,
            "git_revision": runtime.controller.revision,
        }

    @app.get("/api/sessions")
    def sessions() -> list[dict[str, Any]]:
        return runtime.storage.list_sessions()

    @app.post("/api/sessions", status_code=201)
    def create_session(request: SessionCreate) -> dict[str, Any]:
        return runtime.controller.create_session(request.name, request.planes)

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: str) -> dict[str, Any]:
        try:
            return runtime.controller.session_detail(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Sessão não encontrada") from error

    @app.post("/api/sessions/{session_id}/activate")
    def activate_session(session_id: str) -> dict[str, Any]:
        try:
            return runtime.controller.activate_session(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Sessão não encontrada") from error

    @app.post("/api/sessions/{session_id}/plans/{plan_z}/activate")
    def activate_plan(session_id: str, plan_z: float) -> dict[str, Any]:
        try:
            return runtime.controller.activate_plan(session_id, plan_z)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Sessão não encontrada") from error

    @app.post("/api/capture")
    def capture() -> dict[str, Any]:
        return runtime.controller.capture().to_dict()

    @app.post("/api/candidate/decision")
    def candidate_decision(request: Decision) -> dict[str, Any]:
        return runtime.controller.decide(request.confirm)

    @app.post("/api/campaigns", status_code=201)
    def create_campaign(request: CampaignCreate) -> dict[str, Any]:
        view = runtime.campaign.create(request.name, request.square_size_mm)
        assert view is not None
        return view

    @app.post("/api/campaigns/active/checks")
    def campaign_checks(request: CampaignChecks) -> dict[str, Any]:
        return _campaign(runtime.campaign.set_checks(request.model_dump()))

    @app.post("/api/campaigns/active/advance")
    def campaign_advance() -> dict[str, Any]:
        return _campaign(runtime.campaign.advance())

    @app.post("/api/campaigns/active/restart")
    def campaign_restart() -> dict[str, Any]:
        return _campaign(runtime.campaign.restart())

    @app.post("/api/campaigns/active/captures")
    def campaign_capture(request: CaptureDecision) -> dict[str, Any]:
        return _campaign(runtime.campaign.decide_capture(request.decision, request.reason))

    @app.post("/api/campaigns/active/intrinsic/solve")
    def campaign_intrinsic() -> dict[str, Any]:
        return _campaign(runtime.campaign.solve_intrinsic())

    @app.post("/api/campaigns/active/captures/{capture_id}/exclude")
    def campaign_exclude(capture_id: str) -> dict[str, Any]:
        return _campaign(runtime.campaign.exclude_capture(capture_id))

    @app.post("/api/campaigns/active/reprojection/confirm")
    def campaign_reprojection() -> dict[str, Any]:
        return _campaign(runtime.campaign.confirm_reprojection())

    @app.post("/api/campaigns/active/undistort/confirm")
    def campaign_undistort() -> dict[str, Any]:
        return _campaign(runtime.campaign.confirm_undistort())

    @app.post("/api/campaigns/active/frame/{role}")
    def campaign_frame(role: str) -> dict[str, Any]:
        return _campaign(runtime.campaign.record_frame_point(role))

    @app.post("/api/campaigns/active/extrinsic/solve")
    def campaign_extrinsic() -> dict[str, Any]:
        return _campaign(runtime.campaign.solve_pose())

    @app.post("/api/campaigns/active/direction/touch")
    def campaign_touch() -> dict[str, Any]:
        return _campaign(runtime.campaign.record_direction_touch())

    @app.post("/api/campaigns/active/direction/confirm")
    def campaign_direction(request: DirectionConfirm) -> dict[str, Any]:
        return _campaign(runtime.campaign.confirm_direction(request.sense))

    @app.post("/api/campaigns/active/validation/plane")
    def campaign_plane(request: PlaneSelect) -> dict[str, Any]:
        return _campaign(runtime.campaign.select_validation_plane(request.plan_z))

    @app.post("/api/campaigns/active/affine/open")
    def campaign_affine() -> dict[str, Any]:
        return _campaign(runtime.campaign.open_affine_session())

    @app.post("/api/campaigns/active/ransac")
    def campaign_ransac() -> dict[str, Any]:
        return _campaign(runtime.campaign.apply_ransac())

    @app.post("/api/sessions/{session_id}/export")
    def export(session_id: str) -> dict[str, Any]:
        try:
            files = runtime.controller.export_session(session_id, config["app"]["exports_dir"])
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Sessão não encontrada") from error
        return {key: f"/exports/{value}" for key, value in files.items()}

    app.mount(
        "/exports",
        StaticFiles(directory=config["app"]["exports_dir"]),
        name="exports",
    )
    return app
