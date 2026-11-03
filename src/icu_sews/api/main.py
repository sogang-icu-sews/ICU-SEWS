"""FastAPI application entry point. Owner: 김진호."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from icu_sews.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from icu_sews.api.service import InvalidModelInputError, PredictorService

logger = logging.getLogger("uvicorn.error")


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


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    """요청의 처리 결과와 응답 생성까지 걸린 시간을 기록한다."""
    started = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (perf_counter() - started) * 1000

        logger.exception(
            "request_failed method=%s path=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (perf_counter() - started) * 1000

    if response.status_code >= 500:
        log_level = logging.ERROR
    elif response.status_code >= 400:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    logger.log(
        log_level,
        "request_completed method=%s path=%s status=%d elapsed_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report service and model readiness."""
    predictor: PredictorService = request.app.state.predictor
    return HealthResponse(
        status="ok" if predictor.prediction_ready else "degraded",
        model_loaded=predictor.model_loaded,
        mock_mode=predictor.mock_mode,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        503: {"description": "Model unavailable or prediction failed"},
    },
)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """환자 한 명의 관측값으로 예측 결과를 반환한다."""
    predictor: PredictorService = request.app.state.predictor

    try:
        return predictor.predict(payload)

    except InvalidModelInputError as error:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": [
                        "body",
                        "observations",
                        error.observation_index,
                        error.field,
                    ],
                    "msg": str(error),
                    "type": "invalid_model_measurement",
                }
            ],
        ) from error

    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
