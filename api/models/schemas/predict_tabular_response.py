from pydantic import BaseModel
from typing import List


class PredictTabularResponse(BaseModel):
    predictions: List
