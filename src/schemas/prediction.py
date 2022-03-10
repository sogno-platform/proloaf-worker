from typing import Optional, Dict, Any, Union
from enum import Enum
from uuid import UUID
from datetime import datetime
from pydantic import Field, BaseModel
from .data import InputData,TimeseriesData
from .job import Job

# Prediction
class PredictionBase(BaseModel):
    input_data: InputData
    prediction_horizon: datetime
    model_id: int

class PredictionResult(PredictionBase):
    output_data: TimeseriesData

class PredictionJob(Job):
    resource: PredictionBase
    result: Optional[PredictionResult]