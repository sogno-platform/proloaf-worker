from __future__ import annotations
from ast import Bytes
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from datetime import datetime
from pydantic import Field, validator, BaseModel
# from proloaf.base import PydConfigurable as BaseModel
from ..data import InputDataFormat
from ..job import Job
from .proloaf_api_models import ModelWrapper


class ModelType(str, Enum):
    # TODO Placeholder
    model1 = "model1"
    model2 = "model2"


class PredModelBase(BaseModel):
    name: Optional[str]
    model_type: ModelType
    model: Optional[ModelWrapper]


class PredModel(PredModelBase):
    model_id: int
    date_trained: Optional[datetime]
    date_hyperparameter_tuned: Optional[datetime]
    predicted_feature: Optional[str]
    expected_data_format: Optional[InputDataFormat]


class PredModelCreationJob(Job):
    resource: PredModelBase
    result: Optional[PredModel]
