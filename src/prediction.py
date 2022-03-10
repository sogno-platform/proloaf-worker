import json
from typing import Any, ByteString, Dict
import torch
import pandas as pd
import pickle
from .schemas.data import TimeseriesData
from .schemas.job import JobStatus
from .schemas.prediction import PredictionJob, PredictionBase, PredictionResult
from .db import redis_job, redis_model
from proloaf.modelhandler import ModelWrapper


def parse_make_prediction(msg_body: ByteString):
    job = PredictionJob(**json.loads(msg_body))  # , default=str))
    job.status = JobStatus.doing
    redis_job.set(f"pred_{job.job_id}", job.json())
    return job

# XXX this is ProLoaF specific and should be in there
# TODO no length definition yet
def df_to_tensor(df:pd.DataFrame,encoder_features, decoder_features, history_horizon=None, forecast_horizon=None):
    df_enc = df[encoder_features]
    df_dec = df[decoder_features]
    return torch.from_numpy(df_enc.to_numpy()),torch.from_numpy(df_dec.to_numpy())



def handle_make_prediction(baseprediction: PredictionBase):
    pyd_model = pickle.loads(redis_model.get(f"model_{baseprediction.model_id}"))
    model: ModelWrapper = pyd_model.model_object
    df = pd.read_csv("opsd.csv", sep=";")
    
    # TODO add selection of source
    
    forecast = model.predict(df_to_tensor(df, model.encoder_features,model.decoder_features))
    np_forecast = forecast[0].to_numpy()
    pyd_forecast = TimeseriesData(index = range(len(np_forecast)), columns = model.output_labels, data=np_forecast)
    
    return PredictionResult(
        output_data=pyd_forecast
    )
