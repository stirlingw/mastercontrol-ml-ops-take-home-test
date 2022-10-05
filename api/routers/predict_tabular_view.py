from fastapi import APIRouter

from handlers.predict_tabular_handler import predict_tabular_handler
from models.schemas.predict_tabular_request import PredictTabularRequest
from models.schemas.predict_tabular_response import PredictTabularResponse

predict_tabular_view = APIRouter()


@predict_tabular_view.post("/predict_tabular")
async def detect_action(predict_tabular_request: PredictTabularRequest) -> PredictTabularResponse:
    response_json = predict_tabular_handler(predict_tabular_request.dict())
    return PredictTabularResponse(**(response_json))
