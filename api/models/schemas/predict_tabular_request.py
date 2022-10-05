from pydantic import BaseModel


class PredictTabularRequest(BaseModel):
    file_name: str

    # This will appear in the Swagger documentation as the example.
    class Config:
        schema_extra = {
            "example": {
                "file_name": "tht-test-tabular-data.csv",
            }
        }
