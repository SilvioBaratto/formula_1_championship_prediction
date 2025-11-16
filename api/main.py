"""FastAPI server for F1 Championship Prediction API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.simulation import router as simulation_router

app = FastAPI(
    title="F1 Championship Prediction API",
    description="Run Monte Carlo simulations to predict F1 World Championship outcomes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:8080",
        "http://localhost",
        "http://127.0.0.1:4200",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "F1 Championship Prediction API is running",
        "version": "1.0.0",
        "endpoints": {
            "simulate": "/api/v1/simulate",
            "health": "/api/v1/health",
            "drivers": "/api/v1/drivers",
            "standings": "/api/v1/standings",
        },
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "online",
        "message": "F1 Championship Prediction API is healthy",
        "version": "1.0.0",
    }
