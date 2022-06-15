from typing import Any, ByteString, Dict
from sklearn.decomposition import non_negative_factorization
import torch
import pandas as pd
import pickle
from .schemas.data import TimeseriesData
from .schemas.job import JobStatus
from .schemas.prediction import PredictionJob, PredictionBase, PredictionResult
from proloaf.modelhandler import ModelWrapper


def parse_make_prediction(json_message_body, job_db=None):
    job = PredictionJob(**json_message_body)  # , default=str))
    job.status = JobStatus.doing
    if job_db is not None:
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
    prediction_start=None,
):  
    print(df.index)
    df_enc = df[encoder_features].loc[history_horizon:prediction_start]
    df_dec = df[decoder_features].loc[prediction_start:forecast_horizon]
    return torch.from_numpy(df_enc.to_numpy()).float().unsqueeze(
        dim=0
    ), torch.from_numpy(df_dec.to_numpy()).float().unsqueeze(dim=0)


def handle_make_prediction(baseprediction: PredictionBase, model_db=None):
    if isinstance(baseprediction.model, int):
        model_id = baseprediction.model
    else:
        raise NotImplementedError("model can only be accessed via id for now.")
    pyd_model = pickle.loads(model_db.get(f"model_{model_id}"))
    print(f"model_{model_id}")
    print(f"{pyd_model.model.initialized = }")
    model: ModelWrapper = pyd_model.model
    df = pd.read_csv("./proloaf-worker/opsd.csv", sep=";", index_col="Time",parse_dates=True)
    # XXX think of a better way to handle timezone unaware data (idealy there should be no timezone unaware data in the database)
    df.index.tz_localize(tz='utc')
    # TODO add selection of source

    forecast = model.predict(
        *df_to_tensor(
            df,
            model.encoder_features,
            model.decoder_features,
            history_horizon=baseprediction.history_horizon,
            forecast_horizon=baseprediction.forecast_horizon,
            prediction_start=baseprediction.prediction_start,
        )
    )
    print(f"{forecast.size() = }")
    np_forecast = forecast[0].detach().cpu().numpy()
    pyd_forecast = TimeseriesData(
        index=list(range(len(np_forecast))),
        columns=model.output_labels,
        data=np_forecast.tolist(),
    )
    return PredictionResult(**baseprediction.get_config(), output_data=pyd_forecast)
