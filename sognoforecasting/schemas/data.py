from typing import Optional, Dict, Any, Union, List
from enum import Enum
from datetime import datetime
from pydantic import Field #, BaseModel
from proloaf.base import PydConfigurable as BaseModel

# from app.core.base_model import BaseModel

# XXX different basemodel due to redefiniton in api
class TimeseriesData(BaseModel):
    """Input data model.

    The input data request model is base on pandas.DataFrame.to_json() when using
    orient="split". This makes it really convenient to convert to and from a pandas
    DataFrame.
    """

    index: Union[List[int],List[datetime]] = Field(
        ...,
        description="Datetime index of load and features",
        # TODO better example
        example=[
            "2021-04-30T14:00:00.000Z",
            "2021-04-30T14:15:00.000Z",
            "2021-04-30T14:30:00.000Z",
        ],
    
    )
    columns: List[str] = Field(
        ...,
        description="The names of the columns",
        example=["load", "feature_one", "feature_two"],
    )
    data: List[List[Union[float, int, str, None]]] = Field(
        ...,
        description="The data of the columns",
        # TODO better example
        example=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    )


class OutDataDefinition(BaseModel):
    length: int
    resolution_minutes: Optional[int] = Field(60, gt=0,description="Dataresolution in minutes")
    feature_names: List[str] = Field(..., description="The features (previously) used to train the model",
        example=["windspeed", "radiation", "is_saturday", "T-7d"],)


class InputDataFormat(BaseModel):
    max_length: Optional[int] = Field(None, ge=0, description="Maxmium length of a single sample put into a model")
    min_length: Optional[int] = Field(None, ge=0, description="Minimum length of a single sample put into a model")
    resolution_minutes: Optional[int] = Field(60, gt=0,description="Dataresolution in minutes")
    feature_names: List[str] = Field(..., description="The features (previously) used to train the model",
        example=["windspeed", "radiation", "is_saturday", "T-7d"],)


class InputData(BaseModel):
    id: int
    format: Optional[InputDataFormat]


