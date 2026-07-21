import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from app.api.routes import router
from app.api.clinical_routes import router as clinical_router
from app.core.config import get_settings
from app.core.exceptions import MedBriefError
from app.core.logging import configure_logging
from app.db.base import Base
from app.db.session import engine
import app.models  # noqa: F401

configure_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Alembic migrations should replace this convenience hook in production deployment.
        Base.metadata.create_all(bind=engine)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        yield

    application = FastAPI(title=settings.app_name, version="0.1.0",
                          description="OCR, laboratory, and MRI backend services for MedBrief AI",
                          lifespan=lifespan)

    @application.exception_handler(MedBriefError)
    async def domain_error(_: Request, exc: MedBriefError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code,
                            content={"error": exc.code, "message": exc.message})

    @application.exception_handler(IntegrityError)
    async def integrity_error(_: Request, exc: IntegrityError) -> JSONResponse:
        logger.warning("Database constraint violation: %s", exc)
        return JSONResponse(status_code=409,
                            content={"error": "conflict", "message": "Resource already exists"})

    application.include_router(router)
    application.include_router(clinical_router)
    return application


app = create_app()
