from pydantic import BaseModel
from typing import List


class PredictTextResponse(BaseModel):
    vocab: List
    predictions: List
    confidences: List
