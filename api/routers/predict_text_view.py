from fastapi import APIRouter
from models.schemas.predict_text_request import PredictTextRequest
from models.schemas.predict_text_response import PredictTextResponse
from handlers.predict_text_handler import predict_text_handler


predict_text_view = APIRouter()


@predict_text_view.post("/predict_text")
async def detect_action(text_predict_request: PredictTextRequest) -> PredictTextResponse:
    response_json = predict_text_handler(text_predict_request.dict())
    return PredictTextResponse(**(response_json))
