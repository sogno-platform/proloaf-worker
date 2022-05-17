import json
from typing import Any, ByteString, Dict
import torch
import pandas as pd
import pickle
from .schemas.data import TimeseriesData
from .schemas.job import JobStatus
from .schemas.prediction import PredictionJob, PredictionBase, PredictionResult
from .db import dummy_job_db as job_db  # TODO replace with real database redis_job
from .db import (
    dummy_model_db as model_db,
)  # TODO replace with real database redis_model
from .db import dummy_get_unique_id as get_unique_id  # TODO get_unique_id
from proloaf.modelhandler import ModelWrapper


def parse_make_prediction(json_message_body):
    job = PredictionJob(**json_message_body)  # , default=str))
    job.status = JobStatus.doing
    job_db.set(f"{job.job_id}", job.json())
    return job


# XXX this is ProLoaF specific and should be in there
# TODO no length definition yet
def df_to_tensor(
    df: pd.DataFrame,
    encoder_features,
    decoder_features,
    history_horizon=None,
    forecast_horizon=None,
):
    df_enc = df[encoder_features]
    df_dec = df[decoder_features]
    return torch.from_numpy(df_enc.to_numpy()).float().unsqueeze(
        dim=0
    ), torch.from_numpy(df_dec.to_numpy()).float().unsqueeze(dim=0)


def handle_make_prediction(baseprediction: PredictionBase):
    if isinstance(baseprediction.model, int):
        model_id = baseprediction.model
    else:
        raise NotImplementedError("model can only be accessed via id for now.")
    pyd_model = model_db.get(f"model_{model_id}")
    print(f"model_{model_id}")
    print(f"{pyd_model.model.initialized = }")
    model: ModelWrapper = pyd_model.model
    df = pd.read_csv("opsd.csv", sep=";")

    # TODO add selection of source

    forecast = model.predict(
        *df_to_tensor(df, model.encoder_features, model.decoder_features)
    )
    print(f"{forecast.size() = }")
    np_forecast = forecast[0].detach().cpu().numpy()
    pyd_forecast = TimeseriesData(
        index=list(range(len(np_forecast))),
        columns=model.output_labels,
        data=np_forecast.tolist(),
    )

    return PredictionResult(**baseprediction.get_config() ,output_data=pyd_forecast)
