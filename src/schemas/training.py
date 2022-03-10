from __future__ import annotations
from typing import Optional, Dict, Any, Union

from pydantic import Field, BaseModel
from .data import InputData
from .job import Job
from .model import PredModelBase

# Training
class TrainingBase(BaseModel):
    data: InputData
    max_training_time: Optional[int] = Field(
        ..., ge=0, description="Maximum training duration in seconds"
    )
    model: Union[int, PredModelBase] = Field(
        ...,
        description="ID of an existing model or definition for a new one that is to be trained",
    )
    training_configuration: Optional[ProloafTrainingDefinition]


class TrainingResult(TrainingBase):
    actual_training_time: Optional[int] = Field(
        ..., ge=0, description="Actual time spend on training duration in seconds"
    )
    validation_error: Optional[float]
    training_error: float


class TrainingJob(Job):
    resource: TrainingBase
    result: Optional[TrainingResult]


# TODO
class ProloafTrainingDefinition(BaseModel):
    target_id: str = "load"
    optimizer_name: str = "adam"
    early_stopping_patience: int = Field(
        7,
        ge=0,
        description="Number of epochs without improvement before stopping training",
    )
    early_stopping_margin: float = Field(
        0.0, description="Minimum improvement to be considered in early stopping"
    )
    learning_rate: float = 1e-4
    max_epochs: int = Field(
        100, ge=0, description="Maximum number of cycles through the whole training set"
    )
    history_horizon: int = Field(
        24, gt=0, description="Number of timesteps in the past"
    )
    forecast_horizon: int = Field(
        24, gt=0, description="Number of timesteps in the future"
    )
    batch_size: Optional[int] = Field(
        None, gt=0, description="Size of batches, deafults to one batch of maximum size"
    )
