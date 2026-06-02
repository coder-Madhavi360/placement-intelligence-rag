from fastapi import APIRouter, Depends

from core.config import Settings, get_settings
from core.schemas.health import HealthResponse, VersionResponse

router = APIRouter()


@router.get("", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return a lightweight service health payload."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
        version=settings.app_version,
    )


@router.get("/version", response_model=VersionResponse)
async def version(settings: Settings = Depends(get_settings)) -> VersionResponse:
    """Return app version and runtime environment metadata."""
    return VersionResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        api_prefix=settings.api_v1_prefix,
    )
