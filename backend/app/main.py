from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import logger
from app.api import routes_predict, routes_health

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router, prefix=settings.api_v1_prefix, tags=["health"])
app.include_router(routes_predict.router, prefix=settings.api_v1_prefix, tags=["prediction"])


@app.on_event("startup")
async def startup_event():
    logger.info(f"{settings.app_name} starting up...")


@app.get("/")
def root():
    return {"message": settings.app_name, "docs": "/docs"}
