"""
Main FastAPI application for LPBF density prediction
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from api.routes import router
from api.dependencies import get_model, get_feature_schema, get_model_info
from src.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI application

    Loads model and resources on startup, cleans up on shutdown.
    """
    logger.info("=" * 70)
    logger.info("Starting LPBF Density Prediction API")
    logger.info("=" * 70)

    try:
        logger.info("Preloading model...")
        model = get_model()
        logger.info("Model loaded")

        logger.info("Preloading feature schema...")
        schema = get_feature_schema()
        logger.info(f"Feature schema loaded: {schema['n_features']} features")

        logger.info("Getting model info...")
        info = get_model_info()
        if "error" not in info:
            logger.info(f"Model: {info.get('model_name')} v{info.get('version')}")
            logger.info(f"Stage: {info.get('stage')}")
        else:
            logger.warning(f" Could not load model info: {info.get('error')}")

        logger.info("=" * 70)
        logger.info("API Ready to Serve Predictions!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

    yield

    logger.info("Shutting down LPBF Density Prediction API")


app = FastAPI(
    title="LPBF Density Prediction API",
    description="""
    Machine learning API for predicting relative density in Laser Powder Bed Fusion (L-PBF)
    additive manufacturing.

    ## Features
    - **Predict density** based on process parameters and material properties
    - **Model information** about the currently loaded model
    - **Feature ranges** showing valid input ranges from training data
    - **Health check** for monitoring and load balancing

    ## Model
    The API uses a model trained on the Barrionuevo et al. dataset,
    enriched with thermophysical material properties.
    """,
    version="0.1.0",
    contact={
        "name": "BEAM-ML",
        "url": "https://github.com/machin0r/beam-ml",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information"""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Duration: {duration:.3f}s"
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please contact support.",
        },
    )


app.include_router(router, prefix="/api/v1", tags=["Predictions"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LPBF Density Prediction API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting API server...")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
