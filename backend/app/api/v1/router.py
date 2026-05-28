from fastapi import APIRouter
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.api.v1.endpoints import chat, rag
from app.api.v1.routes import health, ingest, query
from app.schemas.health import VersionResponse

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])


@api_router.get("/version", response_model=VersionResponse, tags=["version"])
async def version(settings: Settings = Depends(get_settings)) -> VersionResponse:
    """Return app version and runtime environment metadata."""
    return VersionResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        api_prefix=settings.api_v1_prefix,
    )
