"""FastAPI application entry point. Owner: 김진호."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from icu_sews.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from icu_sews.api.service import PredictorService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the predictor once per server process."""
    app.state.predictor = PredictorService()
    yield


app = FastAPI(
    title="ICU Sepsis Early Warning System API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report service and model readiness."""
    predictor: PredictorService = request.app.state.predictor
    return HealthResponse(
        status="ok" if predictor.model_loaded else "degraded",
        model_loaded=predictor.model_loaded,
        mock_mode=predictor.mock_mode,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Predict one patient's sepsis risk."""
    predictor: PredictorService = request.app.state.predictor
    try:
        return predictor.predict(payload)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

