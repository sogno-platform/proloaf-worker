from typing import Optional, Dict, Any, Union
from enum import Enum
from uuid import UUID
from datetime import datetime
from pydantic import Field #, BaseModel
from proloaf.base import PydConfigurable as BaseModel
from .data import InputData,TimeseriesData
from .job import Job

# Prediction
class PredictionBase(BaseModel):
    history_horizon: datetime #input_data: InputData
    forecast_horizon: datetime
    prediction_start: datetime
    model: int # TODO should this be a Union[int, PredModelBase] like in TrainingBase

class PredictionResult(PredictionBase):
    output_data: TimeseriesData

class PredictionJob(Job):
    resource: PredictionBase
    result: Optional[PredictionResult]