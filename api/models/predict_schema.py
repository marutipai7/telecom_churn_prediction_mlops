from pydantic import BaseModel


class PredictRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: int
