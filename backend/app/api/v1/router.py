from fastapi import APIRouter

from app.api.v1.endpoints import rag
from app.api.v1.routes import health, ingest, query

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
