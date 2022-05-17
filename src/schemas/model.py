from __future__ import annotations
from ast import Bytes
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from datetime import datetime
from pydantic import Field, validator  # , BaseModel
from proloaf.base import PydConfigurable as BaseModel
from .data import InputDataFormat
from .job import Job
from proloaf.modelhandler import ModelWrapper
from proloaf.base import PydConfigurable

BaseModel = PydConfigurable


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


# TODO ProLoaF specific definitions should be imported not defined here


class ProloafModelType(str, Enum):
    RECURRENT = "recurrent"
    SIMPLE_TRANSFORMER = "simple_transformer"


class ProloafRecurrentModelParameters(BaseModel):
    core_net: str = "torch.nn.LSTM"
    core_layers: int = Field(1, gt=0)
    dropout_fc: float = Field(0.4, ge=0, le=1)
    dropout_core: float = Field(0.3, ge=0, le=1)
    rel_linear_hidden_size: float = 1.0
    rel_core_hidden_size: float = 1.0
    relu_leak = 0.1


class ProloafSimpleTransformerModelParameters(BaseModel):
    num_layers: int = Field(3, ge=1)
    dropout: float = Field(0.0, ge=0, le=1)
    n_heads: int = Field(6, ge=1)


# def _default_model_definiton():
#     return {"recurrent": ProloafRecurrentModelParameters(),"simple_transformer":ProloafSimpleTransformerModelParameters()}

# class ProloafModelDefinition(BaseModel):
#     model_class: ProloafModelType = "recurrent"
#     name: Optional[str]
#     training_metric: str = "nllgauss"
#     metric_options: Dict[str,Any] = Field(default_factory=dict)
#     model_parameters: Dict[
#         ProloafModelType,
#         Union[ProloafRecurrentModelParameters, ProloafSimpleTransformerModelParameters],
#     ] = Field(default_factory=_default_model_definiton)
#     encoder_features: List[str] = None,
#     decoder_features: List[str] = None,
# TODO validate dict ({"recurrent": <RecurentDefiniton>, ... })
# TODO most of this should be optional after a default is defined
