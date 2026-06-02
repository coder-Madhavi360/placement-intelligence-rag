from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness payload for load balancers and smoke tests."""

    status: str
    app_name: str
    environment: str
    version: str


class VersionResponse(BaseModel):
    """Build/runtime version payload."""

    app_name: str
    version: str
    environment: str
    api_prefix: str


