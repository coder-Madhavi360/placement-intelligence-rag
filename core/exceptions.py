import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """Standard error payload returned by API exception handlers."""

    detail: str
    status_code: int
    errors: list[dict[str, object]] = Field(default_factory=list)


class ApplicationError(RuntimeError):
    """Base class for predictable application failures."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Application error."

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.detail)
        self.detail = detail or self.detail


class ServiceUnavailableError(ApplicationError):
    """Raised when a configured runtime dependency is unavailable."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Required service is unavailable."


def register_exception_handlers(app: FastAPI) -> None:
    """Attach centralized HTTP, validation, and fallback exception handlers."""

    @app.exception_handler(ApplicationError)
    async def application_error_handler(request: Request, exc: ApplicationError) -> JSONResponse:
        logger.error(
            "Application error: method=%s path=%s status_code=%s detail=%s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return _error_response(exc.status_code, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(
            "Request validation failed: method=%s path=%s errors=%s",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Request validation failed.",
            errors=[dict(error) for error in exc.errors()],
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        logger.warning(
            "HTTP error: method=%s path=%s status_code=%s detail=%s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return _error_response(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: method=%s path=%s", request.method, request.url.path)
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error.",
        )


def _error_response(status_code: int, detail: str, errors: list[dict[str, object]] | None = None) -> JSONResponse:
    payload = ErrorResponse(status_code=status_code, detail=detail, errors=errors or [])
    return JSONResponse(status_code=status_code, content=payload.model_dump())


